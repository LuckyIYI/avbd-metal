import AVBDCore
import Foundation
import simd

public struct HumanoidBoxPhysicalFlowConfiguration: Sendable {
    public var populationSize: Int
    public var proposalProbeSize: Int
    public var generations: Int
    public var maximumWarmupSteps: Int
    public var contactDwellSteps: Int
    public var targetGenerationSteps: Int
    public var targetExecutionSteps: Int?
    public var targetSelectionStep: Int?
    public var candidateTrajectoryDurationSteps: Int?
    public var targetDiscoveryPopulationSize: Int
    public var targetDiscoveryGenerations: Int
    public var targetDiscoveryInitialStandardDeviation: Float
    /// Preserve the complete replay schema while restricting target discovery
    /// to the arm spline and arm-asymmetry heads. This is useful for contact
    /// repair: inherited leg and torso residuals remain physically identical
    /// instead of becoming irrelevant high-dimensional CEM noise.
    public var targetDiscoveryArmOnly: Bool
    /// Preserve the certified grasp spline while restricting target discovery
    /// to leg blending, leg residuals, and torso residuals. Contact-schedule
    /// curricula should not spend samples rewriting a grasp that already
    /// passes exact replay.
    public var targetDiscoveryLowerBodyOnly: Bool
    /// Broadcast each sampled symmetric-arm and arm-asymmetry knot across its
    /// complete spline during target discovery. Short contact-repair stages
    /// can therefore search one coherent eight-DoF correction instead of an
    /// unnecessarily high-dimensional time sequence.
    public var targetDiscoveryTiedArmKnots: Bool
    /// Alternate lower-body, upper-body, and joint CEM generations. This
    /// preserves coupling through exact whole-scene rollouts while avoiding a
    /// single high-dimensional covariance update having to discover weight
    /// transfer and grasp repair simultaneously.
    public var targetDiscoveryBlockCoordinateSearch: Bool
    public var recedingHorizonSteps: Int
    /// Phase used by the first MPC solve when a new physical-flow stage
    /// branches from the middle of an already verified spline. Later replans
    /// start at phase zero. This preserves the remaining controls of a source
    /// plan instead of restarting its action timing at the branch boundary.
    public var recedingInitialPhaseStep: Int
    /// Number of controls from an exactly validated MPC plan to execute
    /// before solving again. A value above one is the standard MPC control
    /// horizon: it prevents a receding optimizer from repeatedly postponing
    /// a required contact transition while retaining a shorter prediction
    /// horizon than a fully open-loop rollout.
    public var recedingControlHorizonSteps: Int
    /// Additional predicted controls beyond the committed MPC control
    /// horizon that must remain physically safe. Nil preserves the strict
    /// full-horizon invariant. A finite value enables standard MPC operation:
    /// only a certified prefix is applied, and the simulator replans before
    /// reaching an unsafe imagined suffix.
    public var recedingSafetyLookaheadSteps: Int?
    /// Extra simulator controls evaluated after the optimized horizon while
    /// holding its terminal residual. This is a terminal-invariance probe:
    /// a candidate cannot commit a short safe prefix that immediately leaves
    /// the recoverable set one control beyond the horizon.
    public var recedingTerminalHoldSteps: Int
    /// Top batched CEM proposals replayed identically across every replica
    /// before a control may be committed. This separates fast proposal
    /// ranking from exact robust branch verification.
    public var recedingValidationCandidateCount: Int
    /// Number of canonical source replicas used to verify each selected MPC
    /// proposal. Nil preserves the historical behavior of reusing the entire
    /// candidate-search batch; setting this explicitly decouples search
    /// population from validation severity.
    public var recedingValidationReplicaCount: Int?
    public var recedingValidationMinimumSuccessFraction: Float
    /// Top reconstruction proposals replayed identically across canonical
    /// replicas before they may become the optimizer incumbent.
    public var reconstructionValidationCandidateCount: Int
    public var reconstructionValidationMinimumSuccessFraction: Float
    /// Rank reconstruction proposals with the same one-environment body and
    /// contact layout used by Policy Replay. Batched replicas are still used
    /// as a separate acceptance gate, so enabling this mode changes proposal
    /// generation rather than weakening the final certificate.
    public var reconstructionOneEnvironmentSearch: Bool
    /// Continue a newly optimized spline from the terminal structured action
    /// of the final certified source stage. Every legacy spline has an
    /// implicit zero knot at its start; without this boundary a composed stage
    /// restarts arm/leg residuals and can drop a carried object before the new
    /// controller has acted.
    public var continueTrajectoryFromSourceTerminal: Bool
    /// Keep the supplied upper-body knots when resuming a trajectory that was
    /// already evaluated from this exact source boundary. Ordinary new
    /// continuations still replace arbitrary arm seeds with the certified
    /// terminal grasp; this opt-in preserves an explicit simulator-derived
    /// repair incumbent while boundary blending keeps its first action
    /// continuous.
    public var preserveProvidedContinuationSeedUpperBody: Bool
    /// Shift a continuation seed by one MPC control so its first proposed
    /// action matches the uncommitted second action from the previous solve.
    public var recedingShiftInitialSeed: Bool
    public var recedingLocomotionBlendProposal: Float?
    /// Make the dedicated locomotion proposal a clean learned-gait candidate
    /// by clearing inherited leg and torso residual knots. The ordinary CEM
    /// mean still retains those residuals, so enabling this adds a distinct
    /// proposal rather than deleting recovery behavior from the search.
    public var recedingLocomotionZeroResidualProposal: Bool
    public var recedingLocomotionCheckpointDirectory: String?
    public var recedingLocomotionCommandSpeed: Float?
    public var recedingForwardOnlyBaseCommand: Bool
    public var recedingHolonomicBaseCommand: Bool
    public var carryBaseLegActionFractionOverride: Float?
    public var legBlendKnotCount: Int
    public var legResidualKnotCount: Int
    public var maximumLegResidualAction: Float
    public var torsoResidualKnotCount: Int
    public var maximumTorsoResidualAction: Float
    public var armAsymmetryKnotCount: Int
    public var maximumArmAsymmetryAction: Float
    /// Blend an explicit object-relative grasp feedback proposal into the
    /// two arm action groups. The target transforms are captured from the
    /// exact physical source state and serialized with the resulting stage;
    /// zero leaves legacy joint-space planning unchanged.
    public var graspAnchorFeedbackBlend: Float
    /// Task-space velocity error is projected this many seconds ahead when
    /// producing the residual arm correction.
    public var graspAnchorFeedbackVelocityHorizonSeconds: Float
    /// Maximum normalized action residual contributed by the grasp feedback
    /// head per arm joint, before the blend and final actuator-range clamp.
    public var graspAnchorFeedbackMaximumActionCorrection: Float
    /// Move each captured hand anchor toward the object's local center by
    /// this distance to maintain bilateral normal-force preload.
    public var graspAnchorFeedbackInwardPreloadMeters: Float
    public var minimumTargetCarryDistanceMeters: Float
    /// Maximum temporary regression below the terminal carry-distance goal
    /// allowed while evaluating a receding-horizon path. Loaded locomotion is
    /// periodic: the box can move backward by millimetres during weight
    /// transfer before advancing at touchdown. The terminal goal remains
    /// unchanged; this only separates recoverable path retention from final
    /// task progress.
    public var maximumTargetPathCarryRegressionMeters: Float
    public var targetDiscoveryObjectiveCarryDistanceMeters: Float?
    /// Required reduction in the measured box-to-receiving-table distance
    /// over this flow segment. Unlike carry distance, this is directional:
    /// moving the load sideways or away from the destination cannot pass it.
    public var minimumTargetDestinationProgressMeters: Float
    public var targetDiscoveryObjectiveDestinationProgressMeters: Float?
    /// Required segment-relative pelvis progress toward the receiving-table
    /// approach pose. This prevents arm-only object motion from being
    /// certified as locomotion.
    public var minimumTargetRootDestinationProgressMeters: Float
    public var targetDiscoveryObjectiveRootDestinationProgressMeters: Float?
    /// Post-swing leading-foot touchdowns added during this segment. One
    /// touchdown is the first hybrid locomotion milestone; alternating steps
    /// below require the following opposite-foot touchdown as well.
    public var minimumTargetTouchdowns: Int
    /// Number of alternating physical touchdowns added during this segment.
    /// The carry task counts a touchdown only after >100 ms of measured swing
    /// and requires the opposite foot from the preceding touchdown.
    public var minimumTargetAlternatingSteps: Int
    /// Continuous discovery signal leading into the discrete touchdown gate:
    /// maximum collision-geometry ground clearance reached by a force-unloaded
    /// physical foot.
    public var minimumTargetSwingFootLiftMeters: Float
    public var targetDiscoveryObjectiveSwingFootLiftMeters: Float?
    /// Required uninterrupted solver-confirmed air time for either foot while
    /// carrying. This supplies a continuous hybrid-contact milestone between
    /// planted balance and the discrete touchdown certificate.
    public var minimumTargetFootAirTimeSeconds: Float
    public var targetDiscoveryObjectiveFootAirTimeSeconds: Float?
    /// Maximum share of total foot normal load carried by one foot. Equal
    /// double support is 0.5; a fully unloaded swing foot approaches 1.0.
    public var minimumTargetFootUnloadingFraction: Float
    public var targetDiscoveryObjectiveFootUnloadingFraction: Float?
    /// Optional endpoint requirement distinct from the historical maximum
    /// above. Continuation artifacts use this to guarantee that the reusable
    /// terminal state is still weight-shifted, rather than merely having
    /// crossed the unloading threshold earlier in the segment.
    public var minimumTargetTerminalFootUnloadingFraction: Float?
    public var minimumTargetClearanceMeters: Float
    public var targetDiscoveryObjectiveClearanceMeters: Float?
    /// Optional viability bound applied at every predicted and committed
    /// control. A terminal-only velocity gate can accidentally prefer a box
    /// impact because the support collision arrests the load at the endpoint.
    /// This path bound exposes unsafe descent before contact.
    public var maximumTargetPathDownwardBoxVelocityMPS: Float?
    /// Optional terminal viability set for receding-horizon control. The
    /// ordinary clearance floor remains hard at every predicted control;
    /// these bounds prevent committing a short safe prefix whose endpoint is
    /// already falling too quickly or has no clearance reserve.
    public var minimumTargetTerminalClearanceMeters: Float?
    public var maximumTargetTerminalDownwardBoxVelocityMPS: Float?
    /// Minimum current opposing-face grasp quality at a certified endpoint.
    /// Zero disables the requirement for legacy balance experiments.
    public var minimumTargetGraspQuality: Float
    public var targetDiscoveryObjectiveGraspQuality: Float?
    public var targetFeasibilityDwellSteps: Int
    public var requireStableCarryPath: Bool
    public var trajectoryKnotCount: Int
    public var robustReplayCount: Int
    public var robustActionNoiseStandardDeviation: Float
    public var optimizationActionNoiseStandardDeviation: Float
    public var optimizationActionNoiseReplicaCount: Int
    public var initialStandardDeviation: Float
    public var eliteFraction: Float
    public var seed: UInt64
    public var optimizerSeed: UInt64?

    public init(
        populationSize: Int = 128,
        proposalProbeSize: Int = 32,
        generations: Int = 6,
        maximumWarmupSteps: Int = 600,
        contactDwellSteps: Int = 8,
        targetGenerationSteps: Int = 400,
        targetExecutionSteps: Int? = nil,
        targetSelectionStep: Int? = nil,
        candidateTrajectoryDurationSteps: Int? = nil,
        targetDiscoveryPopulationSize: Int = 0,
        targetDiscoveryGenerations: Int = 0,
        targetDiscoveryInitialStandardDeviation: Float = 0.25,
        targetDiscoveryArmOnly: Bool = false,
        targetDiscoveryLowerBodyOnly: Bool = false,
        targetDiscoveryTiedArmKnots: Bool = false,
        targetDiscoveryBlockCoordinateSearch: Bool = false,
        recedingHorizonSteps: Int = 0,
        recedingInitialPhaseStep: Int = 0,
        recedingControlHorizonSteps: Int = 1,
        recedingSafetyLookaheadSteps: Int? = nil,
        recedingTerminalHoldSteps: Int = 0,
        recedingValidationCandidateCount: Int = 4,
        recedingValidationReplicaCount: Int? = nil,
        recedingValidationMinimumSuccessFraction: Float = 0.8,
        reconstructionValidationCandidateCount: Int = 4,
        reconstructionValidationMinimumSuccessFraction: Float = 0.8,
        reconstructionOneEnvironmentSearch: Bool = false,
        continueTrajectoryFromSourceTerminal: Bool = false,
        preserveProvidedContinuationSeedUpperBody: Bool = false,
        recedingShiftInitialSeed: Bool = false,
        recedingLocomotionBlendProposal: Float? = nil,
        recedingLocomotionZeroResidualProposal: Bool = false,
        recedingLocomotionCheckpointDirectory: String? = nil,
        recedingLocomotionCommandSpeed: Float? = nil,
        recedingForwardOnlyBaseCommand: Bool = false,
        recedingHolonomicBaseCommand: Bool = false,
        carryBaseLegActionFractionOverride: Float? = nil,
        legBlendKnotCount: Int = 0,
        legResidualKnotCount: Int = 0,
        maximumLegResidualAction: Float = 0.25,
        torsoResidualKnotCount: Int = 0,
        maximumTorsoResidualAction: Float = 0.25,
        armAsymmetryKnotCount: Int = 0,
        maximumArmAsymmetryAction: Float = 0.25,
        graspAnchorFeedbackBlend: Float = 0,
        graspAnchorFeedbackVelocityHorizonSeconds: Float = 0.04,
        graspAnchorFeedbackMaximumActionCorrection: Float = 0.7,
        graspAnchorFeedbackInwardPreloadMeters: Float = 0,
        minimumTargetCarryDistanceMeters: Float = 0,
        maximumTargetPathCarryRegressionMeters: Float = 0,
        targetDiscoveryObjectiveCarryDistanceMeters: Float? = nil,
        minimumTargetDestinationProgressMeters: Float = 0,
        targetDiscoveryObjectiveDestinationProgressMeters: Float? = nil,
        minimumTargetRootDestinationProgressMeters: Float = 0,
        targetDiscoveryObjectiveRootDestinationProgressMeters: Float? = nil,
        minimumTargetTouchdowns: Int = 0,
        minimumTargetAlternatingSteps: Int = 0,
        minimumTargetSwingFootLiftMeters: Float = 0,
        targetDiscoveryObjectiveSwingFootLiftMeters: Float? = nil,
        minimumTargetFootAirTimeSeconds: Float = 0,
        targetDiscoveryObjectiveFootAirTimeSeconds: Float? = nil,
        minimumTargetFootUnloadingFraction: Float = 0,
        targetDiscoveryObjectiveFootUnloadingFraction: Float? = nil,
        minimumTargetTerminalFootUnloadingFraction: Float? = nil,
        minimumTargetClearanceMeters: Float = 0.01,
        targetDiscoveryObjectiveClearanceMeters: Float? = nil,
        maximumTargetPathDownwardBoxVelocityMPS: Float? = nil,
        minimumTargetTerminalClearanceMeters: Float? = nil,
        maximumTargetTerminalDownwardBoxVelocityMPS: Float? = nil,
        minimumTargetGraspQuality: Float = 0,
        targetDiscoveryObjectiveGraspQuality: Float? = nil,
        targetFeasibilityDwellSteps: Int = 1,
        requireStableCarryPath: Bool = false,
        trajectoryKnotCount: Int = 5,
        robustReplayCount: Int = 32,
        robustActionNoiseStandardDeviation: Float = 0,
        optimizationActionNoiseStandardDeviation: Float = 0,
        optimizationActionNoiseReplicaCount: Int = 1,
        initialStandardDeviation: Float = 0.25,
        eliteFraction: Float = 0.05,
        seed: UInt64 = 1,
        optimizerSeed: UInt64? = nil
    ) {
        self.populationSize = populationSize
        self.proposalProbeSize = proposalProbeSize
        self.generations = generations
        self.maximumWarmupSteps = maximumWarmupSteps
        self.contactDwellSteps = contactDwellSteps
        self.targetGenerationSteps = targetGenerationSteps
        self.targetExecutionSteps = targetExecutionSteps
        self.targetSelectionStep = targetSelectionStep
        self.candidateTrajectoryDurationSteps =
            candidateTrajectoryDurationSteps
        self.targetDiscoveryPopulationSize = targetDiscoveryPopulationSize
        self.targetDiscoveryGenerations = targetDiscoveryGenerations
        self.targetDiscoveryInitialStandardDeviation =
            targetDiscoveryInitialStandardDeviation
        self.targetDiscoveryArmOnly = targetDiscoveryArmOnly
        self.targetDiscoveryLowerBodyOnly =
            targetDiscoveryLowerBodyOnly
        self.targetDiscoveryTiedArmKnots =
            targetDiscoveryTiedArmKnots
        self.targetDiscoveryBlockCoordinateSearch =
            targetDiscoveryBlockCoordinateSearch
        self.recedingHorizonSteps = recedingHorizonSteps
        self.recedingInitialPhaseStep = recedingInitialPhaseStep
        self.recedingControlHorizonSteps = recedingControlHorizonSteps
        self.recedingSafetyLookaheadSteps =
            recedingSafetyLookaheadSteps
        self.recedingTerminalHoldSteps = recedingTerminalHoldSteps
        self.recedingValidationCandidateCount =
            recedingValidationCandidateCount
        self.recedingValidationReplicaCount =
            recedingValidationReplicaCount
        self.recedingValidationMinimumSuccessFraction =
            recedingValidationMinimumSuccessFraction
        self.reconstructionValidationCandidateCount =
            reconstructionValidationCandidateCount
        self.reconstructionValidationMinimumSuccessFraction =
            reconstructionValidationMinimumSuccessFraction
        self.reconstructionOneEnvironmentSearch =
            reconstructionOneEnvironmentSearch
        self.continueTrajectoryFromSourceTerminal =
            continueTrajectoryFromSourceTerminal
        self.preserveProvidedContinuationSeedUpperBody =
            preserveProvidedContinuationSeedUpperBody
        self.recedingShiftInitialSeed = recedingShiftInitialSeed
        self.recedingLocomotionBlendProposal =
            recedingLocomotionBlendProposal
        self.recedingLocomotionZeroResidualProposal =
            recedingLocomotionZeroResidualProposal
        self.recedingLocomotionCheckpointDirectory =
            recedingLocomotionCheckpointDirectory
        self.recedingLocomotionCommandSpeed =
            recedingLocomotionCommandSpeed
        self.recedingForwardOnlyBaseCommand =
            recedingForwardOnlyBaseCommand
        self.recedingHolonomicBaseCommand =
            recedingHolonomicBaseCommand
        self.carryBaseLegActionFractionOverride =
            carryBaseLegActionFractionOverride
        self.legBlendKnotCount = legBlendKnotCount
        self.legResidualKnotCount = legResidualKnotCount
        self.maximumLegResidualAction = maximumLegResidualAction
        self.torsoResidualKnotCount = torsoResidualKnotCount
        self.maximumTorsoResidualAction = maximumTorsoResidualAction
        self.armAsymmetryKnotCount = armAsymmetryKnotCount
        self.maximumArmAsymmetryAction = maximumArmAsymmetryAction
        self.graspAnchorFeedbackBlend = graspAnchorFeedbackBlend
        self.graspAnchorFeedbackVelocityHorizonSeconds =
            graspAnchorFeedbackVelocityHorizonSeconds
        self.graspAnchorFeedbackMaximumActionCorrection =
            graspAnchorFeedbackMaximumActionCorrection
        self.graspAnchorFeedbackInwardPreloadMeters =
            graspAnchorFeedbackInwardPreloadMeters
        self.minimumTargetCarryDistanceMeters =
            minimumTargetCarryDistanceMeters
        self.maximumTargetPathCarryRegressionMeters =
            maximumTargetPathCarryRegressionMeters
        self.targetDiscoveryObjectiveCarryDistanceMeters =
            targetDiscoveryObjectiveCarryDistanceMeters
        self.minimumTargetDestinationProgressMeters =
            minimumTargetDestinationProgressMeters
        self.targetDiscoveryObjectiveDestinationProgressMeters =
            targetDiscoveryObjectiveDestinationProgressMeters
        self.minimumTargetRootDestinationProgressMeters =
            minimumTargetRootDestinationProgressMeters
        self.targetDiscoveryObjectiveRootDestinationProgressMeters =
            targetDiscoveryObjectiveRootDestinationProgressMeters
        self.minimumTargetTouchdowns = minimumTargetTouchdowns
        self.minimumTargetAlternatingSteps =
            minimumTargetAlternatingSteps
        self.minimumTargetSwingFootLiftMeters =
            minimumTargetSwingFootLiftMeters
        self.targetDiscoveryObjectiveSwingFootLiftMeters =
            targetDiscoveryObjectiveSwingFootLiftMeters
        self.minimumTargetFootAirTimeSeconds =
            minimumTargetFootAirTimeSeconds
        self.targetDiscoveryObjectiveFootAirTimeSeconds =
            targetDiscoveryObjectiveFootAirTimeSeconds
        self.minimumTargetFootUnloadingFraction =
            minimumTargetFootUnloadingFraction
        self.targetDiscoveryObjectiveFootUnloadingFraction =
            targetDiscoveryObjectiveFootUnloadingFraction
        self.minimumTargetTerminalFootUnloadingFraction =
            minimumTargetTerminalFootUnloadingFraction
        self.minimumTargetClearanceMeters = minimumTargetClearanceMeters
        self.targetDiscoveryObjectiveClearanceMeters =
            targetDiscoveryObjectiveClearanceMeters
        self.maximumTargetPathDownwardBoxVelocityMPS =
            maximumTargetPathDownwardBoxVelocityMPS
        self.minimumTargetTerminalClearanceMeters =
            minimumTargetTerminalClearanceMeters
        self.maximumTargetTerminalDownwardBoxVelocityMPS =
            maximumTargetTerminalDownwardBoxVelocityMPS
        self.minimumTargetGraspQuality = minimumTargetGraspQuality
        self.targetDiscoveryObjectiveGraspQuality =
            targetDiscoveryObjectiveGraspQuality
        self.targetFeasibilityDwellSteps = targetFeasibilityDwellSteps
        self.requireStableCarryPath = requireStableCarryPath
        self.trajectoryKnotCount = trajectoryKnotCount
        self.robustReplayCount = robustReplayCount
        self.robustActionNoiseStandardDeviation =
            robustActionNoiseStandardDeviation
        self.optimizationActionNoiseStandardDeviation =
            optimizationActionNoiseStandardDeviation
        self.optimizationActionNoiseReplicaCount =
            optimizationActionNoiseReplicaCount
        self.initialStandardDeviation = initialStandardDeviation
        self.eliteFraction = eliteFraction
        self.seed = seed
        self.optimizerSeed = optimizerSeed
    }

    func validate() throws {
        guard populationSize >= 8,
              proposalProbeSize >= 2,
              proposalProbeSize <= populationSize,
              generations > 0,
              maximumWarmupSteps > 0,
              contactDwellSteps > 0,
              targetGenerationSteps >= 8,
              (targetExecutionSteps.map {
                  $0 > 0 && $0 <= targetGenerationSteps
              } ?? true),
              (targetSelectionStep.map {
                  $0 > 0
                    && $0 <= (targetExecutionSteps
                        ?? targetGenerationSteps)
              } ?? true),
              (candidateTrajectoryDurationSteps.map { $0 > 0 } ?? true),
              ((targetDiscoveryPopulationSize == 0
                    && targetDiscoveryGenerations == 0)
                || (targetDiscoveryPopulationSize >= 8
                    && targetDiscoveryGenerations > 0)),
              targetDiscoveryInitialStandardDeviation.isFinite,
              targetDiscoveryInitialStandardDeviation > 0,
              (!targetDiscoveryArmOnly
                || targetDiscoveryPopulationSize >= 8),
              (!targetDiscoveryLowerBodyOnly
                || targetDiscoveryPopulationSize >= 8),
              (!targetDiscoveryTiedArmKnots
                || targetDiscoveryPopulationSize >= 8),
              (!targetDiscoveryBlockCoordinateSearch
                || (recedingHorizonSteps > 0
                    && !targetDiscoveryArmOnly
                    && !targetDiscoveryLowerBodyOnly)),
              !(targetDiscoveryArmOnly && targetDiscoveryLowerBodyOnly),
              recedingHorizonSteps >= 0,
              recedingInitialPhaseStep >= 0,
              (recedingHorizonSteps > 0
                ? recedingInitialPhaseStep < recedingHorizonSteps
                : recedingInitialPhaseStep == 0),
              recedingControlHorizonSteps > 0,
              (recedingHorizonSteps > 0
                ? recedingControlHorizonSteps <= recedingHorizonSteps
                : recedingControlHorizonSteps == 1),
              (recedingSafetyLookaheadSteps.map {
                  $0 >= 0 && recedingHorizonSteps > 0
              } ?? true),
              recedingTerminalHoldSteps >= 0,
              recedingValidationCandidateCount > 0,
              recedingHorizonSteps == 0
                || recedingValidationCandidateCount
                    <= targetDiscoveryPopulationSize,
              (recedingValidationReplicaCount.map {
                  recedingHorizonSteps > 0 && $0 > 0
                    && $0 <= targetDiscoveryPopulationSize
              } ?? true),
              recedingValidationMinimumSuccessFraction.isFinite,
              (0.5...1).contains(
                recedingValidationMinimumSuccessFraction),
              reconstructionValidationCandidateCount >= 2,
              reconstructionValidationCandidateCount <= populationSize,
              reconstructionValidationMinimumSuccessFraction.isFinite,
              (0.5...1).contains(
                reconstructionValidationMinimumSuccessFraction),
              (recedingTerminalHoldSteps == 0
                || recedingHorizonSteps > 0),
              (!preserveProvidedContinuationSeedUpperBody
                || continueTrajectoryFromSourceTerminal),
              (recedingHorizonSteps == 0
                || (targetDiscoveryPopulationSize >= 8
                    && targetDiscoveryGenerations > 0)),
              (recedingHorizonSteps == 0
                || targetFeasibilityDwellSteps <= recedingHorizonSteps),
              (recedingLocomotionBlendProposal.map {
                  $0.isFinite && (0...1).contains($0)
                    && legBlendKnotCount > 0
              } ?? true),
              (!recedingLocomotionZeroResidualProposal
                || recedingLocomotionBlendProposal != nil),
              (recedingLocomotionCheckpointDirectory.map {
                  !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && recedingHorizonSteps > 0
                    && legBlendKnotCount > 0
              } ?? true),
              (recedingLocomotionCommandSpeed.map {
                  $0.isFinite && $0 > 0 && $0 <= 1
                    && recedingLocomotionCheckpointDirectory != nil
                    && (recedingForwardOnlyBaseCommand
                        != recedingHolonomicBaseCommand)
              } ?? true),
              !(recedingForwardOnlyBaseCommand
                    && recedingHolonomicBaseCommand),
              (carryBaseLegActionFractionOverride.map {
                  $0.isFinite && (0...1).contains($0)
              } ?? true),
              legBlendKnotCount >= 0,
              legResidualKnotCount >= 0,
              maximumLegResidualAction.isFinite,
              maximumLegResidualAction > 0,
              maximumLegResidualAction <= 1,
              torsoResidualKnotCount >= 0,
              maximumTorsoResidualAction.isFinite,
              maximumTorsoResidualAction > 0,
              maximumTorsoResidualAction <= 1,
              armAsymmetryKnotCount >= 0,
              maximumArmAsymmetryAction.isFinite,
              maximumArmAsymmetryAction > 0,
              maximumArmAsymmetryAction <= 1,
              graspAnchorFeedbackBlend.isFinite,
              (0...1).contains(graspAnchorFeedbackBlend),
              graspAnchorFeedbackVelocityHorizonSeconds.isFinite,
              (0...0.2).contains(
                graspAnchorFeedbackVelocityHorizonSeconds),
              graspAnchorFeedbackMaximumActionCorrection.isFinite,
              graspAnchorFeedbackMaximumActionCorrection > 0,
              graspAnchorFeedbackMaximumActionCorrection <= 1,
              graspAnchorFeedbackInwardPreloadMeters.isFinite,
              (0...0.03).contains(
                graspAnchorFeedbackInwardPreloadMeters),
              minimumTargetCarryDistanceMeters.isFinite,
              minimumTargetCarryDistanceMeters >= 0,
              maximumTargetPathCarryRegressionMeters.isFinite,
              maximumTargetPathCarryRegressionMeters >= 0,
              maximumTargetPathCarryRegressionMeters
                <= minimumTargetCarryDistanceMeters,
              (targetDiscoveryObjectiveCarryDistanceMeters.map {
                  $0.isFinite
                    && $0 >= minimumTargetCarryDistanceMeters
              } ?? true),
              minimumTargetDestinationProgressMeters.isFinite,
              minimumTargetDestinationProgressMeters >= 0,
              (targetDiscoveryObjectiveDestinationProgressMeters.map {
                  $0.isFinite
                    && $0 >= minimumTargetDestinationProgressMeters
              } ?? true),
              minimumTargetRootDestinationProgressMeters.isFinite,
              minimumTargetRootDestinationProgressMeters >= 0,
              (targetDiscoveryObjectiveRootDestinationProgressMeters.map {
                  $0.isFinite
                    && $0 >= minimumTargetRootDestinationProgressMeters
              } ?? true),
              minimumTargetTouchdowns >= 0,
              minimumTargetAlternatingSteps >= 0,
              minimumTargetSwingFootLiftMeters.isFinite,
              minimumTargetSwingFootLiftMeters >= 0,
              minimumTargetSwingFootLiftMeters <= 0.25,
              (targetDiscoveryObjectiveSwingFootLiftMeters.map {
                  $0.isFinite
                    && $0 > 0
                    && $0 >= minimumTargetSwingFootLiftMeters
                    && $0 <= 0.25
              } ?? true),
              minimumTargetFootAirTimeSeconds.isFinite,
              minimumTargetFootAirTimeSeconds >= 0,
              minimumTargetFootAirTimeSeconds <= 1,
              (targetDiscoveryObjectiveFootAirTimeSeconds.map {
                  $0.isFinite
                    && $0 >= minimumTargetFootAirTimeSeconds
                    && $0 <= 1
              } ?? true),
              minimumTargetFootUnloadingFraction.isFinite,
              (0...1).contains(minimumTargetFootUnloadingFraction),
              (targetDiscoveryObjectiveFootUnloadingFraction.map {
                  $0.isFinite
                    && $0 >= minimumTargetFootUnloadingFraction
                    && $0 <= 1
              } ?? true),
              (minimumTargetTerminalFootUnloadingFraction.map {
                  $0.isFinite && (0...1).contains($0)
              } ?? true),
              minimumTargetClearanceMeters.isFinite,
              minimumTargetClearanceMeters >= 0.01,
              (targetDiscoveryObjectiveClearanceMeters.map {
                  $0.isFinite && $0 >= minimumTargetClearanceMeters
              } ?? true),
              (maximumTargetPathDownwardBoxVelocityMPS.map {
                  $0.isFinite && $0 >= 0
              } ?? true),
              (minimumTargetTerminalClearanceMeters.map {
                  $0.isFinite && $0 >= minimumTargetClearanceMeters
              } ?? true),
              (maximumTargetTerminalDownwardBoxVelocityMPS.map {
                  $0.isFinite && $0 >= 0
              } ?? true),
              minimumTargetGraspQuality.isFinite,
              (0...1).contains(minimumTargetGraspQuality),
              (targetDiscoveryObjectiveGraspQuality.map {
                  $0.isFinite && $0 >= minimumTargetGraspQuality && $0 <= 1
              } ?? true),
              targetFeasibilityDwellSteps > 0,
              trajectoryKnotCount > 0,
              robustReplayCount > 0,
              robustActionNoiseStandardDeviation.isFinite,
              robustActionNoiseStandardDeviation >= 0,
              optimizationActionNoiseStandardDeviation.isFinite,
              optimizationActionNoiseStandardDeviation >= 0,
              optimizationActionNoiseReplicaCount > 0,
              initialStandardDeviation.isFinite,
              initialStandardDeviation > 0,
              eliteFraction > 0,
              eliteFraction <= 0.5 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid-box physical-flow configuration")
        }
    }
}

/// One simulator-executed segment in a composable physical-state flow.
/// Keeping the complete ordered lineage in every derived artifact prevents a
/// continuation from being mistaken for a standalone action trajectory.
public struct HumanoidBoxPhysicalFlowStage: Codable, Sendable {
    public var trajectory: [Float]
    public var trajectorySequence: [[Float]]?
    /// Evaluation phase for each receding plan in `trajectorySequence`.
    /// Zero means the first control of that plan. A positive value records a
    /// committed control from a previously validated fallback continuation.
    public var trajectorySequencePhaseSteps: [Int]?
    public var controlSteps: Int
    public var trajectoryDurationSteps: Int?
    public var trajectorySequenceStepDenominator: Int?
    public var forwardOnlyBaseCommand: Bool?
    public var holonomicBaseCommand: Bool?
    public var locomotionCheckpointDirectory: String?
    public var locomotionCommandSpeed: Float?
    public var policyOnly: Bool?
    /// Exact full-body commands committed for this stage. When present these
    /// are the authoritative execution semantics; the structured trajectory
    /// remains as planner lineage and as a warm-start representation.
    public var appliedNormalizedActions: [[Float]]?
    /// Recreate the exact speculative branch boundary from which this stage
    /// was optimized. This is part of the declared physical-flow lineage:
    /// long contact-rich prefixes may leave batched rows with different
    /// hidden solver warm starts, while a reconstructed continuation is
    /// validated from identical cold-start copies of row zero.
    public var canonicalizeReplicasBeforeExecution: Bool?
    /// Replace this stage's implicit zero spline boundary with the terminal
    /// structured action of the preceding certified trajectory stage.
    public var continueFromPreviousTrajectoryTerminal: Bool?
    /// Explicit structured-controller semantics used by this stage. Anchors
    /// are hand-center transforms in the physical box frame.
    public var graspAnchorFeedbackBlend: Float?
    public var graspAnchorFeedbackVelocityHorizonSeconds: Float?
    public var graspAnchorFeedbackMaximumActionCorrection: Float?
    public var graspAnchorFeedbackInwardPreloadMeters: Float?
    public var leftGraspAnchorBoxLocalMeters: [Float]?
    public var rightGraspAnchorBoxLocalMeters: [Float]?
    public var graspAnchorBoxHeightMeters: Float?
    public var minimumCarryDistanceMeters: Float?
    public var minimumDestinationProgressMeters: Float?
    public var minimumRootDestinationProgressMeters: Float?
    public var minimumTouchdowns: Int?
    public var minimumAlternatingSteps: Int?
    public var minimumSwingFootLiftMeters: Float?
    public var minimumFootAirTimeSeconds: Float?
    public var minimumFootUnloadingFraction: Float?
    public var minimumTerminalFootUnloadingFraction: Float?
    public var minimumClearanceMeters: Float?
    /// Maximum permitted downward box speed at every control in this
    /// certified stage. Nil preserves legacy artifacts.
    public var maximumPathDownwardBoxVelocityMPS: Float?
    public var minimumGraspQuality: Float?
    public var certificationDwellSteps: Int?

    public init(
        trajectory: [Float], controlSteps: Int,
        trajectoryDurationSteps: Int? = nil,
        trajectorySequence: [[Float]]? = nil,
        trajectorySequencePhaseSteps: [Int]? = nil,
        trajectorySequenceStepDenominator: Int? = nil,
        forwardOnlyBaseCommand: Bool = false,
        holonomicBaseCommand: Bool = false,
        locomotionCheckpointDirectory: String? = nil,
        locomotionCommandSpeed: Float? = nil,
        policyOnly: Bool = false,
        appliedNormalizedActions: [[Float]]? = nil,
        canonicalizeReplicasBeforeExecution: Bool = false,
        continueFromPreviousTrajectoryTerminal: Bool = false,
        graspAnchorFeedbackBlend: Float? = nil,
        graspAnchorFeedbackVelocityHorizonSeconds: Float? = nil,
        graspAnchorFeedbackMaximumActionCorrection: Float? = nil,
        graspAnchorFeedbackInwardPreloadMeters: Float? = nil,
        leftGraspAnchorBoxLocalMeters: [Float]? = nil,
        rightGraspAnchorBoxLocalMeters: [Float]? = nil,
        graspAnchorBoxHeightMeters: Float? = nil,
        minimumCarryDistanceMeters: Float? = nil,
        minimumDestinationProgressMeters: Float? = nil,
        minimumRootDestinationProgressMeters: Float? = nil,
        minimumTouchdowns: Int? = nil,
        minimumAlternatingSteps: Int? = nil,
        minimumSwingFootLiftMeters: Float? = nil,
        minimumFootAirTimeSeconds: Float? = nil,
        minimumFootUnloadingFraction: Float? = nil,
        minimumTerminalFootUnloadingFraction: Float? = nil,
        minimumClearanceMeters: Float? = nil,
        maximumPathDownwardBoxVelocityMPS: Float? = nil,
        minimumGraspQuality: Float? = nil,
        certificationDwellSteps: Int? = nil
    ) {
        self.trajectory = trajectory
        self.trajectorySequence = trajectorySequence
        self.trajectorySequencePhaseSteps =
            trajectorySequencePhaseSteps
        self.controlSteps = controlSteps
        self.trajectoryDurationSteps = trajectoryDurationSteps
        self.trajectorySequenceStepDenominator =
            trajectorySequenceStepDenominator
        self.forwardOnlyBaseCommand = forwardOnlyBaseCommand ? true : nil
        self.holonomicBaseCommand = holonomicBaseCommand ? true : nil
        self.locomotionCheckpointDirectory = locomotionCheckpointDirectory
        self.locomotionCommandSpeed = locomotionCommandSpeed
        self.policyOnly = policyOnly ? true : nil
        self.appliedNormalizedActions = appliedNormalizedActions
        self.canonicalizeReplicasBeforeExecution =
            canonicalizeReplicasBeforeExecution ? true : nil
        self.continueFromPreviousTrajectoryTerminal =
            continueFromPreviousTrajectoryTerminal ? true : nil
        self.graspAnchorFeedbackBlend = graspAnchorFeedbackBlend
        self.graspAnchorFeedbackVelocityHorizonSeconds =
            graspAnchorFeedbackVelocityHorizonSeconds
        self.graspAnchorFeedbackMaximumActionCorrection =
            graspAnchorFeedbackMaximumActionCorrection
        self.graspAnchorFeedbackInwardPreloadMeters =
            graspAnchorFeedbackInwardPreloadMeters
        self.leftGraspAnchorBoxLocalMeters =
            leftGraspAnchorBoxLocalMeters
        self.rightGraspAnchorBoxLocalMeters =
            rightGraspAnchorBoxLocalMeters
        self.graspAnchorBoxHeightMeters =
            graspAnchorBoxHeightMeters
        self.minimumCarryDistanceMeters = minimumCarryDistanceMeters
        self.minimumDestinationProgressMeters =
            minimumDestinationProgressMeters
        self.minimumRootDestinationProgressMeters =
            minimumRootDestinationProgressMeters
        self.minimumTouchdowns = minimumTouchdowns
        self.minimumAlternatingSteps = minimumAlternatingSteps
        self.minimumSwingFootLiftMeters =
            minimumSwingFootLiftMeters
        self.minimumFootAirTimeSeconds = minimumFootAirTimeSeconds
        self.minimumFootUnloadingFraction =
            minimumFootUnloadingFraction
        self.minimumTerminalFootUnloadingFraction =
            minimumTerminalFootUnloadingFraction
        self.minimumClearanceMeters = minimumClearanceMeters
        self.maximumPathDownwardBoxVelocityMPS =
            maximumPathDownwardBoxVelocityMPS
        self.minimumGraspQuality = minimumGraspQuality
        self.certificationDwellSteps = certificationDwellSteps
    }

    /// Return an exact prefix of this simulator-executed stage. Sequence
    /// stages retain their original interpolation denominator, so the prefix
    /// reproduces the same controls and physical state as the corresponding
    /// point in the longer certified path.
    public func prefix(
        controlSteps requestedControlSteps: Int
    ) -> HumanoidBoxPhysicalFlowStage? {
        guard requestedControlSteps > 0,
              requestedControlSteps <= controlSteps else { return nil }
        var result = self
        result.controlSteps = requestedControlSteps
        if let sequence = trajectorySequence {
            guard sequence.count == controlSteps else { return nil }
            result.trajectorySequence = Array(
                sequence.prefix(requestedControlSteps))
            result.trajectorySequencePhaseSteps =
                trajectorySequencePhaseSteps.map {
                    Array($0.prefix(requestedControlSteps))
                }
        }
        if let actions = appliedNormalizedActions {
            guard actions.count == controlSteps else { return nil }
            result.appliedNormalizedActions = Array(
                actions.prefix(requestedControlSteps))
        }
        return result
    }

    public func trajectoryEvaluationStep(
        at controlStep: Int
    ) -> Int {
        precondition((0..<controlSteps).contains(controlStep))
        guard trajectorySequence != nil else { return controlStep }
        return trajectorySequencePhaseSteps?[controlStep] ?? 0
    }
}

public struct HumanoidBoxPhysicalFlowMetrics: Codable, Sendable {
    public var loss: Float
    public var maximumNormalizedError: Float
    public var rootPositionErrorMeters: Float
    public var rootRotationErrorRadians: Float
    public var rootLinearVelocityErrorMPS: Float
    public var rootAngularVelocityErrorRadPS: Float
    public var jointAngleRMSErrorRadians: Float
    public var jointVelocityRMSErrorRadPS: Float
    public var maximumFootPositionErrorMeters: Float
    public var maximumFootVelocityErrorMPS: Float
    public var boxPositionErrorMeters: Float
    public var boxRotationErrorRadians: Float
    public var boxLinearVelocityErrorMPS: Float
    public var boxAngularVelocityErrorRadPS: Float
    public var maximumHandPositionErrorMeters: Float
    public var maximumHandVelocityErrorMPS: Float
    public var carryDistanceErrorMeters: Float
    public var clearanceMeters: Float
    public var graspQuality: Float?
    public var bilateralHandContact: Bool
    public var unsupported: Bool
    public var physicallyLifted: Bool
    public var robotUpright: Bool
    public var boxUpright: Bool
    public var failed: Bool
    public var minimumCarryDistanceAchieved: Bool
    public var minimumClearanceAchieved: Bool
    public var minimumGraspQualityAchieved: Bool?
    public var feasibilityDwellSteps: Int
    public var minimumDwellAchieved: Bool
    /// Worst smooth prerequisite penalty across the required terminal dwell
    /// window. Zero means every sampled endpoint predicate passed throughout
    /// the complete window; values above four identify a hard-margin miss.
    public var feasibilityWindowMaximumNormalizedError: Float? = nil
    public var stablePathViolationSteps: Int
    public var stablePathAchieved: Bool
    public var terminalFootUnloadingFraction: Float? = nil
    public var minimumTerminalFootUnloadingAchieved: Bool? = nil

    public var endpointPassed: Bool {
        maximumNormalizedError < 1 && minimumCarryDistanceAchieved
            && minimumClearanceAchieved
            && (minimumGraspQualityAchieved ?? true)
            && (minimumTerminalFootUnloadingAchieved ?? true)
            && minimumDwellAchieved
            && stablePathAchieved
    }
}

public struct HumanoidBoxPhysicalFlowGeneration: Codable, Sendable {
    public var generation: Int
    public var bestLoss: Float
    public var medianLoss: Float
    public var bestMaximumNormalizedError: Float
    public var meanStandardDeviation: Float
    public var warmupControlSteps: Int
}

public struct HumanoidBoxPhysicalFlowTraceSample: Codable, Sendable {
    public var step: Int
    public var rootPositionMeters: [Float]
    public var rootLinearVelocityMPS: [Float]
    public var rootUprightAlignment: Float
    public var boxPositionMeters: [Float]
    public var boxLinearVelocityMPS: [Float]
    public var boxUprightAlignment: Float
    public var boxClearanceMeters: Float
    public var carryDistanceMeters: Float
    public var placementDistanceMeters: Float
    public var destinationProgressMeters: Float
    public var rootDestinationProgressMeters: Float? = nil
    public var loadedTouchdowns: Int? = nil
    public var loadedAlternatingSteps: Int? = nil
    public var maximumSwingFootLiftMeters: Float? = nil
    public var maximumLoadedFootAirTimeSeconds: Float? = nil
    public var maximumFootUnloadingFraction: Float? = nil
    public var footUnloadingFraction: Float? = nil
    /// Narrowphase manifold presence is useful for debugging, but it is not
    /// evidence that a sole supports load. Keep both signals explicit.
    public var leftFootContact: Bool? = nil
    public var rightFootContact: Bool? = nil
    public var leftLoadBearingFootContact: Bool? = nil
    public var rightLoadBearingFootContact: Bool? = nil
    public var leftFootNormalLoad: Float? = nil
    public var rightFootNormalLoad: Float? = nil
    public var leftFootGroundClearanceMeters: Float? = nil
    public var rightFootGroundClearanceMeters: Float? = nil
    public var leftLoadedFootAirTimeSeconds: Float? = nil
    public var rightLoadedFootAirTimeSeconds: Float? = nil
    public var maximumActuatorTorqueRatio: Float? = nil
    public var maximumArmActuatorTorqueRatio: Float? = nil
    public var saturatedActuatorCount: Int? = nil
    public var saturatedArmActuatorCount: Int? = nil
    public var minimumJointLimitMarginRadians: Float? = nil
    public var maximumRequestedTargetClampRadians: Float? = nil
    public var boxRootRelativeSpeedMPS: Float? = nil
    public var leftHandContact: Bool
    public var rightHandContact: Bool
    public var leftHandNormalLoad: Float? = nil
    public var rightHandNormalLoad: Float? = nil
    public var graspFrictionSupportFraction: Float? = nil
    public var leftHandRelativeToBoxMeters: [Float]? = nil
    public var rightHandRelativeToBoxMeters: [Float]? = nil
    public var leftHandLinearVelocityMPS: [Float]? = nil
    public var rightHandLinearVelocityMPS: [Float]? = nil
    public var leftHandBoxRelativeSpeedMPS: Float? = nil
    public var rightHandBoxRelativeSpeedMPS: Float? = nil
    /// Task-space residual requested by the object-relative grasp controller
    /// after its Cartesian clamp. These diagnostics make controller
    /// saturation and poor Jacobian projection visible in saved artifacts.
    public var leftGraspFeedbackTaskDeltaMeters: [Float]? = nil
    public var rightGraspFeedbackTaskDeltaMeters: [Float]? = nil
    public var leftGraspFeedbackUnclampedTaskDeltaMagnitudeMeters: Float? = nil
    public var rightGraspFeedbackUnclampedTaskDeltaMagnitudeMeters: Float? = nil
    public var leftGraspFeedbackJointDeltaRadians: [Float]? = nil
    public var rightGraspFeedbackJointDeltaRadians: [Float]? = nil
    public var leftGraspFeedbackActionCorrection: [Float]? = nil
    public var rightGraspFeedbackActionCorrection: [Float]? = nil
    /// Final normalized arm command that was actually submitted for the
    /// transition ending at this sample. Nil for the initial state.
    public var leftAppliedArmActions: [Float]? = nil
    public var rightAppliedArmActions: [Float]? = nil
    /// Reconstructed pre-feedback arm commands. These are exact when the
    /// final composition was not clamped; the value is nil at a clamp bound.
    /// Comparing them with the requested correction exposes cancellation
    /// between a learned/spline command and the grasp-retention layer.
    public var leftInferredPreFeedbackArmActions: [Float]? = nil
    public var rightInferredPreFeedbackArmActions: [Float]? = nil
    public var leftArmCompositionClamped: Bool? = nil
    public var rightArmCompositionClamped: Bool? = nil
    /// Complete normalized policy command actually submitted for the
    /// transition ending at this sample. This is the authoritative
    /// distillation label: it includes locomotion, torso, both arms, and any
    /// state-dependent grasp correction. Nil for the initial state and for
    /// artifacts produced before this field was introduced.
    public var appliedNormalizedActions: [Float]? = nil
    public var leftHandTargetDistanceMeters: Float? = nil
    public var rightHandTargetDistanceMeters: Float? = nil
    public var handOppositionAlignment: Float? = nil
    public var graspQuality: Float
    public var loadBearingGrasp: Bool
    public var boxPedestalContact: Bool
    public var boxDestinationContact: Bool
    public var boxGroundContact: Bool? = nil
    public var unsupported: Bool? = nil
    public var physicallyLifted: Bool
    public var failed: Bool
}

public struct HumanoidBoxPhysicalFlowReport: Codable, Sendable {
    public var experiment: String
    public var checkpointDirectory: String
    public var seed: UInt64
    public var optimizerSeed: UInt64?
    public var populationSize: Int
    public var proposalProbeSize: Int? = nil
    public var generations: Int
    public var targetTrajectoryWithheldFromSearch: Bool
    public var targetGenerationSteps: Int
    public var targetExecutionSteps: Int
    public var candidateTrajectoryDurationSteps: Int?
    public var targetDiscoveryPopulationSize: Int
    public var targetDiscoveryGenerations: Int
    public var targetDiscoveryInitialStandardDeviation: Float? = nil
    public var targetDiscoveryCandidateRollouts: Int
    public var targetDiscoveryArmOnly: Bool? = nil
    public var targetDiscoveryLowerBodyOnly: Bool? = nil
    public var targetDiscoveryTiedArmKnots: Bool? = nil
    public var targetDiscoveryBlockCoordinateSearch: Bool? = nil
    public var targetDiscoverySeededFromSourceTerminalUpperBody: Bool? = nil
    public var targetDiscoveryHeldSourceTerminalUpperBody: Bool? = nil
    public var targetDiscoveryPreservedProvidedUpperBodySeed: Bool? = nil
    public var targetGeneratingTrajectory: [Float]
    public var targetGeneratingTrajectorySequence: [[Float]]?
    public var targetGeneratingTrajectorySequencePhaseSteps: [Int]? = nil
    /// The derived target/reconstruction stage begins from a canonicalized
    /// speculative branch boundary. Persisting this execution semantic keeps
    /// future lineage replay exact while allowing older artifacts to retain
    /// their pre-boundary behavior.
    public var derivedStageSourceCanonicalized: Bool? = true
    public var derivedStageContinuesFromSourceTerminal: Bool? = nil
    public var targetCommittedTrace: [HumanoidBoxPhysicalFlowTraceSample]?
    public var recedingHorizonSteps: Int?
    public var recedingInitialPhaseStep: Int? = nil
    public var recedingControlHorizonSteps: Int? = nil
    public var recedingSafetyLookaheadSteps: Int? = nil
    public var recedingTerminalHoldSteps: Int? = nil
    public var recedingValidationCandidateCount: Int? = nil
    public var recedingValidationReplicaCount: Int? = nil
    public var recedingValidationMinimumSuccessFraction: Float? = nil
    public var reconstructionValidationCandidateCount: Int? = nil
    public var reconstructionValidationMinimumSuccessFraction: Float? = nil
    public var reconstructionValidationSuccessFraction: Float? = nil
    public var reconstructionOneEnvironmentSearch: Bool? = nil
    public var recedingShiftInitialSeed: Bool?
    public var recedingLocomotionBlendProposal: Float?
    public var recedingLocomotionZeroResidualProposal: Bool? = nil
    public var recedingLocomotionCheckpointDirectory: String?
    public var recedingLocomotionCommandSpeed: Float?
    public var recedingForwardOnlyBaseCommand: Bool?
    public var recedingHolonomicBaseCommand: Bool?
    public var carryBaseLegActionFractionOverride: Float?
    public var legBlendKnotCount: Int
    public var legResidualKnotCount: Int
    public var maximumLegResidualAction: Float
    public var torsoResidualKnotCount: Int?
    public var maximumTorsoResidualAction: Float?
    public var armAsymmetryKnotCount: Int?
    public var maximumArmAsymmetryAction: Float?
    public var graspAnchorFeedbackBlend: Float? = nil
    public var graspAnchorFeedbackVelocityHorizonSeconds: Float? = nil
    public var graspAnchorFeedbackMaximumActionCorrection: Float? = nil
    public var graspAnchorFeedbackInwardPreloadMeters: Float? = nil
    public var leftGraspAnchorBoxLocalMeters: [Float]? = nil
    public var rightGraspAnchorBoxLocalMeters: [Float]? = nil
    public var graspAnchorBoxHeightMeters: Float? = nil
    public var minimumTargetCarryDistanceMeters: Float
    public var minimumTargetPathCarryDistanceMeters: Float? = nil
    public var targetDiscoveryObjectiveCarryDistanceMeters: Float
    public var minimumTargetDestinationProgressMeters: Float?
    public var targetDiscoveryObjectiveDestinationProgressMeters: Float?
    public var minimumTargetRootDestinationProgressMeters: Float? = nil
    public var targetDiscoveryObjectiveRootDestinationProgressMeters:
        Float? = nil
    public var minimumTargetTouchdowns: Int? = nil
    public var minimumTargetAlternatingSteps: Int? = nil
    public var minimumTargetSwingFootLiftMeters: Float? = nil
    public var targetDiscoveryObjectiveSwingFootLiftMeters: Float? = nil
    public var minimumTargetFootAirTimeSeconds: Float? = nil
    public var targetDiscoveryObjectiveFootAirTimeSeconds: Float? = nil
    public var minimumTargetFootUnloadingFraction: Float? = nil
    public var targetDiscoveryObjectiveFootUnloadingFraction: Float? = nil
    public var minimumTargetTerminalFootUnloadingFraction: Float? = nil
    public var minimumTargetClearanceMeters: Float
    public var targetDiscoveryObjectiveClearanceMeters: Float
    public var maximumTargetPathDownwardBoxVelocityMPS: Float? = nil
    public var minimumTargetTerminalClearanceMeters: Float? = nil
    public var maximumTargetTerminalDownwardBoxVelocityMPS: Float? = nil
    public var minimumTargetGraspQuality: Float?
    public var targetDiscoveryObjectiveGraspQuality: Float?
    public var targetFeasibilityDwellSteps: Int
    public var requireStableCarryPath: Bool
    public var sourceTrajectorySteps: Int
    public var sourceStages: [HumanoidBoxPhysicalFlowStage]
    /// Policy-derived commands preceding the commissioned source stages.
    /// Their count is the exact grasp-establishment stopping time used by the
    /// physical audit, which otherwise changes with speculative batch width.
    public var sourceWarmupAppliedActions: [[Float]]? = nil
    /// Exact full-body commands used to reach the certified target-start
    /// boundary, flattened across source stages. The source prefix is still
    /// explicitly reported; preserving its actual commands prevents later
    /// learners from silently approximating different controller semantics.
    public var sourceAppliedActions: [[Float]]? = nil
    public var sourceReplaySuccessFraction: Float
    public var selectedTargetStep: Int
    public var targetClearanceMeters: Float
    public var targetGraspQuality: Float?
    public var targetBoxUprightAlignment: Float
    public var targetRobotUprightAlignment: Float
    public var targetCarryDistanceMeters: Float
    public var targetPlacementDistanceMeters: Float?
    public var targetDestinationProgressMeters: Float?
    public var targetRootDestinationProgressMeters: Float? = nil
    public var targetLoadedTouchdowns: Int? = nil
    public var targetLoadedAlternatingSteps: Int? = nil
    public var targetMaximumSwingFootLiftMeters: Float? = nil
    public var targetMaximumLoadedFootAirTimeSeconds: Float? = nil
    public var targetMaximumFootUnloadingFraction: Float? = nil
    public var targetTerminalFootUnloadingFraction: Float? = nil
    public var targetStableCarryPath: Bool
    public var targetPredictedRecoveryPathSafe: Bool? = nil
    public var targetCloneSuccessFraction: Float
    public var targetReplayMaximumNormalizedError: Float
    public var providedProposal: HumanoidBoxPhysicalFlowMetrics?
    public var zeroProposal: HumanoidBoxPhysicalFlowMetrics
    public var generationZeroSelectedProposal:
        PhysicalFlowProposalSelection
    public var providedProposalProbeBestLoss: Float?
    public var zeroProposalProbeBestLoss: Float?
    public var optimized: HumanoidBoxPhysicalFlowMetrics
    public var optimizedToZeroLossRatio: Float
    public var robustReplaySuccessFraction: Float
    public var robustActionNoiseStandardDeviation: Float?
    public var optimizationActionNoiseStandardDeviation: Float?
    public var optimizationActionNoiseReplicaCount: Int?
    public var initialStandardDeviation: Float? = nil
    public var eliteFraction: Float? = nil
    public var robustReplayMedianMaximumNormalizedError: Float
    public var robustReplayWorstMaximumNormalizedError: Float
    public var selectedReplayMaximumNormalizedStateError: Float
    public var bestTrajectory: [Float]
    public var generationHistory: [HumanoidBoxPhysicalFlowGeneration]
    public var candidateRollouts: Int
    public var simulatedEnvironmentControlSteps: Int
    public var elapsedSeconds: Double
    public var infrastructureGatePassed: Bool
    public var targetGatePassed: Bool
    /// The finite target prefix passed its physical endpoint, replica, and
    /// exact-reconstruction checks. This is evidence, but it is not by itself
    /// a safe boundary from which another controller may continue.
    public var targetFinitePrefixGatePassed: Bool? = nil
    public var targetPlanningGatePassed: Bool
    /// The finite-prefix gate plus a recovery-safe terminal state. Only this
    /// gate authorizes the target stage as reusable source lineage.
    public var targetReusableFrontierGatePassed: Bool? = nil
    public var reconstructionGatePassed: Bool
    public var robustReplayGatePassed: Bool
    public var goGatePassed: Bool
}

/// Structured controller value at a trajectory boundary. Persisting the
/// boundary separately from the candidate parameters makes a failed receding
/// plan replayable without guessing which previous spline endpoint it used.
public struct HumanoidBoxPhysicalFlowTrajectoryBoundary: Codable, Equatable,
    Sendable
{
    public var armDelta: [Float]
    public var legBlend: Float
    public var legResidual: [Float]
    public var torsoResidual: Float
    public var armAsymmetry: [Float]

    public init(
        armDelta: [Float], legBlend: Float, legResidual: [Float],
        torsoResidual: Float, armAsymmetry: [Float]
    ) {
        self.armDelta = armDelta
        self.legBlend = legBlend
        self.legResidual = legResidual
        self.torsoResidual = torsoResidual
        self.armAsymmetry = armAsymmetry
    }
}

public struct HumanoidBoxPhysicalFlowTargetFailure: Error, Codable,
    CustomStringConvertible, Sendable
{
    public var experiment: String
    public var seed: UInt64
    public var optimizerSeed: UInt64?
    public var targetGeneratingTrajectory: [Float]
    public var targetGeneratingTrajectorySequence: [[Float]]?
    public var targetGeneratingTrajectorySequencePhaseSteps: [Int]? = nil
    public var derivedStageSourceCanonicalized: Bool? = true
    public var derivedStageContinuesFromSourceTerminal: Bool? = nil
    public var committedTrace: [HumanoidBoxPhysicalFlowTraceSample]? = nil
    public var targetDiscoveryPopulationSize: Int? = nil
    public var targetDiscoveryGenerations: Int? = nil
    public var targetDiscoveryCandidateRollouts: Int? = nil
    public var targetDiscoveryTiedArmKnots: Bool? = nil
    public var targetDiscoveryBlockCoordinateSearch: Bool? = nil
    public var targetDiscoverySeededFromSourceTerminalUpperBody: Bool? = nil
    public var targetDiscoveryHeldSourceTerminalUpperBody: Bool? = nil
    public var targetDiscoveryPreservedProvidedUpperBodySeed: Bool? = nil
    public var recedingValidationMinimumSuccessFraction: Float? = nil
    public var nearMissLoss: Float? = nil
    public var nearMissValidationSuccessFraction: Float? = nil
    public var nearMissFirstControlSafe: Bool? = nil
    public var nearMissCommitPathSafe: Bool? = nil
    public var nearMissPredictedPathSafe: Bool? = nil
    public var nearMissTerminalGoalFeasible: Bool? = nil
    public var nearMissTerminalRecoveryViable: Bool? = nil
    public var nearMissParameters: [Float]? = nil
    public var nearMissPhaseStep: Int? = nil
    public var nearMissTrajectoryStart:
        HumanoidBoxPhysicalFlowTrajectoryBoundary? = nil
    public var nearMissPredictedTrace:
        [HumanoidBoxPhysicalFlowTraceSample]? = nil
    public var bestStableSwingLoss: Float? = nil
    public var bestStableSwingTerminalGoalFeasible: Bool? = nil
    public var bestStableSwingTerminalRecoveryViable: Bool? = nil
    public var bestStableSwingParameters: [Float]? = nil
    public var bestStableSwingTrace:
        [HumanoidBoxPhysicalFlowTraceSample]? = nil
    public var bestSwingFrontierLoss: Float? = nil
    public var bestSwingFrontierFirstControlSafe: Bool? = nil
    public var bestSwingFrontierPredictedPathSafe: Bool? = nil
    public var bestSwingFrontierTerminalGoalFeasible: Bool? = nil
    public var bestSwingFrontierTerminalRecoveryViable: Bool? = nil
    public var bestSwingFrontierParameters: [Float]? = nil
    public var bestSwingFrontierTrace:
        [HumanoidBoxPhysicalFlowTraceSample]? = nil
    public var bestFeasibilityFrontierLoss: Float? = nil
    public var bestFeasibilityFrontierDwellSteps: Int? = nil
    public var bestFeasibilityFrontierTerminalGoalFeasible: Bool? = nil
    public var bestFeasibilityFrontierTerminalGoalComponents:
        [String: Bool]? = nil
    public var bestFeasibilityFrontierTerminalRecoveryViable: Bool? = nil
    public var bestFeasibilityFrontierParameters: [Float]? = nil
    public var bestFeasibilityFrontierTrace:
        [HumanoidBoxPhysicalFlowTraceSample]? = nil
    public var recedingValidationReplicaCount: Int? = nil
    public var recedingLocomotionCheckpointDirectory: String? = nil
    public var recedingLocomotionCommandSpeed: Float? = nil
    public var recedingLocomotionBlendProposal: Float? = nil
    public var recedingLocomotionZeroResidualProposal: Bool? = nil
    public var recedingForwardOnlyBaseCommand: Bool? = nil
    public var recedingHolonomicBaseCommand: Bool? = nil
    public var targetGenerationSteps: Int
    public var recedingInitialPhaseStep: Int? = nil
    public var recedingControlHorizonSteps: Int? = nil
    public var recedingSafetyLookaheadSteps: Int? = nil
    public var recedingTerminalHoldSteps: Int? = nil
    public var targetExecutionSteps: Int? = nil
    public var sourceStages: [HumanoidBoxPhysicalFlowStage] = []
    /// Exact source lineage captured on first execution. Failure artifacts
    /// must retain it because batch-dependent warm-up termination cannot be
    /// reconstructed from a checkpoint and seed alone.
    public var sourceWarmupAppliedActions: [[Float]]? = nil
    public var sourceAppliedActions: [[Float]]? = nil
    public var legBlendKnotCount: Int
    public var legResidualKnotCount: Int
    public var maximumLegResidualAction: Float?
    public var torsoResidualKnotCount: Int
    public var maximumTorsoResidualAction: Float?
    public var armAsymmetryKnotCount: Int?
    public var maximumArmAsymmetryAction: Float?
    public var graspAnchorFeedbackBlend: Float? = nil
    public var graspAnchorFeedbackVelocityHorizonSeconds: Float? = nil
    public var graspAnchorFeedbackMaximumActionCorrection: Float? = nil
    public var graspAnchorFeedbackInwardPreloadMeters: Float? = nil
    public var leftGraspAnchorBoxLocalMeters: [Float]? = nil
    public var rightGraspAnchorBoxLocalMeters: [Float]? = nil
    public var graspAnchorBoxHeightMeters: Float? = nil
    public var maximumClearanceMeters: Float
    public var maximumCarryDistanceMeters: Float
    public var maximumStableCarryDistanceMeters: Float
    public var maximumDestinationProgressMeters: Float? = nil
    public var maximumRootDestinationProgressMeters: Float? = nil
    public var maximumLoadedTouchdowns: Int? = nil
    public var maximumLoadedAlternatingSteps: Int? = nil
    public var maximumSwingFootLiftMeters: Float? = nil
    public var maximumLoadedFootAirTimeSeconds: Float? = nil
    public var maximumFootUnloadingFraction: Float? = nil
    public var maximumFeasibilityDwellSteps: Int
    public var maximumBilateralDwellSteps: Int?
    public var maximumLoadBearingGraspDwellSteps: Int? = nil
    public var maximumUnsupportedDwellSteps: Int?
    public var maximumPhysicallyLiftedDwellSteps: Int?
    public var maximumUprightDwellSteps: Int?
    public var maximumCarryThresholdDwellSteps: Int?
    public var maximumClearanceThresholdDwellSteps: Int?
    public var maximumGraspQualityThresholdDwellSteps: Int? = nil
    public var maximumGraspQuality: Float? = nil
    public var firstStablePathViolationStep: Int?
    public var finalCarryDistanceMeters: Float? = nil
    public var finalPlacementDistanceMeters: Float? = nil
    public var finalDestinationProgressMeters: Float? = nil
    public var finalRootDestinationProgressMeters: Float? = nil
    public var finalLoadedTouchdowns: Int? = nil
    public var finalLoadedAlternatingSteps: Int? = nil
    public var finalClearanceMeters: Float? = nil
    public var finalGraspQuality: Float? = nil
    public var finalBoxVerticalVelocityMPS: Float? = nil
    /// A useful goal-independent certificate for training recovery/hold
    /// policies. It requires the complete executed path to remain upright,
    /// bilateral, unsupported, physically lifted, and at least 1 cm clear;
    /// it does not relabel failed carry-distance tracking as transport.
    public var physicalBalanceGatePassed: Bool? = nil
    public var requiredCarryDistanceMeters: Float
    public var requiredPathCarryDistanceMeters: Float? = nil
    public var requiredDestinationProgressMeters: Float? = nil
    public var requiredRootDestinationProgressMeters: Float? = nil
    public var requiredTouchdowns: Int? = nil
    public var requiredAlternatingSteps: Int? = nil
    public var requiredSwingFootLiftMeters: Float? = nil
    public var objectiveSwingFootLiftMeters: Float? = nil
    public var requiredFootAirTimeSeconds: Float? = nil
    public var objectiveFootAirTimeSeconds: Float? = nil
    public var requiredFootUnloadingFraction: Float? = nil
    public var objectiveFootUnloadingFraction: Float? = nil
    public var requiredTerminalFootUnloadingFraction: Float? = nil
    public var requiredClearanceMeters: Float
    public var maximumPathDownwardBoxVelocityMPS: Float? = nil
    public var requiredTerminalClearanceMeters: Float? = nil
    public var maximumTerminalDownwardBoxVelocityMPS: Float? = nil
    public var requiredGraspQuality: Float? = nil
    public var requiredFeasibilityDwellSteps: Int
    public var targetPlanningGatePassed: Bool = false
    public var goGatePassed: Bool = false

    public var description: String {
        String(format:
            "hidden trajectory produced no upright unsupported target (max clearance %.4f m, max carry %.4f m, max stable carry %.4f m, max root progress %.4f m, max touchdowns %d, max alternating steps %d, max foot lift %.4f m, max foot air %.3f s, max foot unload %.3f, max grasp %.4f, max dwell %d/%d, bilateral %d, load-bearing %d, unsupported %d, lifted %d, upright %d, carry %d, clearance %d, grasp %d, first path violation %d, required carry %.4f m, required root progress %.4f m, required touchdowns %d, required steps %d, required foot lift %.4f m, required foot air %.3f s, required foot unload %.3f, required clearance %.4f m, required grasp %.4f)",
            maximumClearanceMeters, maximumCarryDistanceMeters,
            maximumStableCarryDistanceMeters,
            maximumRootDestinationProgressMeters ?? -1,
            maximumLoadedTouchdowns ?? -1,
            maximumLoadedAlternatingSteps ?? -1,
            maximumSwingFootLiftMeters ?? -1,
            maximumLoadedFootAirTimeSeconds ?? -1,
            maximumFootUnloadingFraction ?? -1,
            maximumGraspQuality ?? -1,
            maximumFeasibilityDwellSteps,
            requiredFeasibilityDwellSteps,
            maximumBilateralDwellSteps ?? -1,
            maximumLoadBearingGraspDwellSteps ?? -1,
            maximumUnsupportedDwellSteps ?? -1,
            maximumPhysicallyLiftedDwellSteps ?? -1,
            maximumUprightDwellSteps ?? -1,
            maximumCarryThresholdDwellSteps ?? -1,
            maximumClearanceThresholdDwellSteps ?? -1,
            maximumGraspQualityThresholdDwellSteps ?? -1,
            firstStablePathViolationStep ?? -1,
            requiredCarryDistanceMeters,
            requiredRootDestinationProgressMeters ?? 0,
            requiredTouchdowns ?? 0,
            requiredAlternatingSteps ?? 0,
            requiredSwingFootLiftMeters ?? 0,
            requiredFootAirTimeSeconds ?? 0,
            requiredFootUnloadingFraction ?? 0,
            requiredClearanceMeters,
            requiredGraspQuality ?? 0)
    }
}

/// First humanoid scaling bridge for the physical-flow controller. A hidden
/// trajectory creates a real, unsupported H1+box target state. Search receives
/// only that state plus an unrelated stored trajectory as a proposal; every
/// candidate and final replay is executed by the simulator.
public enum HumanoidBoxPhysicalFlowExperiment {
    private static let firstArmAction = 11
    private static let armActionCount = 8

    typealias StructuredTrajectoryBoundary =
        HumanoidBoxPhysicalFlowTrajectoryBoundary

    struct StructuredTrajectorySequenceBoundaryReplay {
        var starts: [StructuredTrajectoryBoundary?]
        var terminal: StructuredTrajectoryBoundary?
    }

    private struct ObjectRelativeGraspFeedback {
        var blend: Float
        var velocityHorizonSeconds: Float
        var maximumActionCorrection: Float
        var inwardPreloadMeters: Float
        var leftAnchorBoxLocal: F3
        var rightAnchorBoxLocal: F3
        var boxHeightMeters: Float
    }

    private struct GraspFeedbackArmDiagnostic {
        var taskDelta: F3
        var unclampedTaskDeltaMagnitude: Float
        var jointDeltaRadians: [Float]
        var normalizedActionCorrection: [Float]
    }

    struct RobustValidationSummary: Equatable {
        var requiredSuccessCount: Int
        var successFraction: Float
        var replicaAgreementPassed: Bool
        var exactBranchPassed: Bool
        var passed: Bool
        var lowerQuantileIndex: Int
        var upperQuantileIndex: Int
    }

    /// Convert a fractional robustness contract into exact finite-batch
    /// semantics. The first row is the branch committed to the task and shown
    /// in Policy Replay, so consensus may reject it but never overrule it.
    static func robustValidationSummary(
        replicaPasses: [Bool],
        minimumSuccessFraction: Float
    ) -> RobustValidationSummary {
        precondition(!replicaPasses.isEmpty)
        precondition(minimumSuccessFraction.isFinite)
        precondition((0...1).contains(minimumSuccessFraction))
        let requiredCount = min(
            replicaPasses.count,
            max(1, Int(ceil(
                minimumSuccessFraction * Float(replicaPasses.count)))))
        let successCount = replicaPasses.lazy.filter { $0 }.count
        let successFraction = Float(successCount)
            / Float(replicaPasses.count)
        let replicaAgreementPassed =
            successCount >= requiredCount
        let exactBranchPassed = replicaPasses[0]
        return RobustValidationSummary(
            requiredSuccessCount: requiredCount,
            successFraction: successFraction,
            replicaAgreementPassed: replicaAgreementPassed,
            exactBranchPassed: exactBranchPassed,
            passed: replicaAgreementPassed && exactBranchPassed,
            lowerQuantileIndex: replicaPasses.count - requiredCount,
            upperQuantileIndex: requiredCount - 1)
    }

    static func reconstructionEliteCount(
        populationSize: Int,
        eliteFraction: Float,
        validatedCandidateCount: Int
    ) -> Int {
        precondition(populationSize >= 2)
        precondition(eliteFraction.isFinite && eliteFraction > 0)
        precondition(validatedCandidateCount >= 2)
        return min(
            validatedCandidateCount,
            max(2, Int(Float(populationSize) * eliteFraction)))
    }

    static func continuedTrajectoryValue(
        zeroStartedValue: Float,
        initialValue: Float,
        progress: Float,
        knotCount: Int
    ) -> Float {
        precondition(zeroStartedValue.isFinite && initialValue.isFinite)
        precondition(progress.isFinite && knotCount > 0)
        let initialWeight = max(
            1 - simd_clamp(progress, 0, 1) * Float(knotCount), 0)
        return zeroStartedValue + initialWeight * initialValue
    }

    static func structuredTrajectoryBoundary(
        _ parameters: [Float],
        progress: Float,
        armKnotCount: Int,
        blendKnotCount: Int,
        legResidualKnotCount: Int,
        maximumLegResidualAction: Float,
        torsoResidualKnotCount: Int,
        maximumTorsoResidualAction: Float,
        armAsymmetryKnotCount: Int,
        maximumArmAsymmetryAction: Float
    ) -> StructuredTrajectoryBoundary {
        let armParameterCount = 4 * armKnotCount
        let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            Array(parameters.prefix(armParameterCount)),
            knotCount: armKnotCount,
            progress: progress)
        let blend = blendKnotCount > 0
            ? legBlendFraction(
                parameters, progress: progress,
                armParameterCount: armParameterCount,
                knotCount: blendKnotCount)
            : 0
        let legs = legResidualKnotCount > 0
            ? (0..<10).map {
                legResidualAction(
                    parameters, action: $0, progress: progress,
                    armParameterCount: armParameterCount,
                    blendKnotCount: blendKnotCount,
                    residualKnotCount: legResidualKnotCount,
                    maximumAction: maximumLegResidualAction)
            } : [Float](repeating: 0, count: 10)
        let torso = torsoResidualKnotCount > 0
            ? torsoResidualAction(
                parameters, progress: progress,
                armParameterCount: armParameterCount,
                blendKnotCount: blendKnotCount,
                legResidualKnotCount: legResidualKnotCount,
                torsoResidualKnotCount: torsoResidualKnotCount,
                maximumAction: maximumTorsoResidualAction)
            : 0
        let asymmetry = armAsymmetryKnotCount > 0
            ? (0..<4).map {
                armAsymmetryAction(
                    parameters, action: $0, progress: progress,
                    armParameterCount: armParameterCount,
                    blendKnotCount: blendKnotCount,
                    legResidualKnotCount: legResidualKnotCount,
                    torsoResidualKnotCount: torsoResidualKnotCount,
                    asymmetryKnotCount: armAsymmetryKnotCount,
                    maximumAction: maximumArmAsymmetryAction)
            } : [Float](repeating: 0, count: 4)
        return StructuredTrajectoryBoundary(
            armDelta: arm,
            legBlend: blend,
            legResidual: legs,
            torsoResidual: torso,
            armAsymmetry: asymmetry)
    }

    static func continuedStructuredTrajectoryBoundary(
        _ parameters: [Float],
        progress: Float,
        initial: StructuredTrajectoryBoundary?,
        armKnotCount: Int,
        blendKnotCount: Int,
        legResidualKnotCount: Int,
        maximumLegResidualAction: Float,
        torsoResidualKnotCount: Int,
        maximumTorsoResidualAction: Float,
        armAsymmetryKnotCount: Int,
        maximumArmAsymmetryAction: Float
    ) -> StructuredTrajectoryBoundary {
        var boundary = structuredTrajectoryBoundary(
            parameters,
            progress: progress,
            armKnotCount: armKnotCount,
            blendKnotCount: blendKnotCount,
            legResidualKnotCount: legResidualKnotCount,
            maximumLegResidualAction: maximumLegResidualAction,
            torsoResidualKnotCount: torsoResidualKnotCount,
            maximumTorsoResidualAction: maximumTorsoResidualAction,
            armAsymmetryKnotCount: armAsymmetryKnotCount,
            maximumArmAsymmetryAction: maximumArmAsymmetryAction)
        guard let initial else { return boundary }
        for action in boundary.armDelta.indices {
            boundary.armDelta[action] = continuedTrajectoryValue(
                zeroStartedValue: boundary.armDelta[action],
                initialValue: initial.armDelta[action],
                progress: progress,
                knotCount: armKnotCount)
        }
        if blendKnotCount > 0 {
            boundary.legBlend = continuedTrajectoryValue(
                zeroStartedValue: boundary.legBlend,
                initialValue: initial.legBlend,
                progress: progress,
                knotCount: blendKnotCount)
        }
        if legResidualKnotCount > 0 {
            for action in boundary.legResidual.indices {
                boundary.legResidual[action] = continuedTrajectoryValue(
                    zeroStartedValue: boundary.legResidual[action],
                    initialValue: initial.legResidual[action],
                    progress: progress,
                    knotCount: legResidualKnotCount)
            }
        }
        if torsoResidualKnotCount > 0 {
            boundary.torsoResidual = continuedTrajectoryValue(
                zeroStartedValue: boundary.torsoResidual,
                initialValue: initial.torsoResidual,
                progress: progress,
                knotCount: torsoResidualKnotCount)
        }
        if armAsymmetryKnotCount > 0 {
            for action in boundary.armAsymmetry.indices {
                boundary.armAsymmetry[action] = continuedTrajectoryValue(
                    zeroStartedValue: boundary.armAsymmetry[action],
                    initialValue: initial.armAsymmetry[action],
                    progress: progress,
                    knotCount: armAsymmetryKnotCount)
            }
        }
        return boundary
    }

    /// Reconstruct the controller boundary used by every row of a serialized
    /// receding-horizon trajectory. Consecutive increasing phases with the
    /// same parameters are controls from one already-validated plan and must
    /// retain that plan's original boundary. A phase reset or parameter change
    /// starts a new plan from the previously executed control's terminal
    /// boundary.
    static func structuredTrajectorySequenceBoundaryReplay(
        _ sequence: [[Float]],
        phaseSteps: [Int]?,
        denominator: Int,
        initial: StructuredTrajectoryBoundary?,
        armKnotCount: Int,
        blendKnotCount: Int,
        legResidualKnotCount: Int,
        maximumLegResidualAction: Float,
        torsoResidualKnotCount: Int,
        maximumTorsoResidualAction: Float,
        armAsymmetryKnotCount: Int,
        maximumArmAsymmetryAction: Float
    ) -> StructuredTrajectorySequenceBoundaryReplay {
        precondition(!sequence.isEmpty && denominator > 0)
        precondition(phaseSteps?.count == sequence.count || phaseSteps == nil)
        var starts = [StructuredTrajectoryBoundary?]()
        starts.reserveCapacity(sequence.count)
        var activePlanStart = initial
        var terminal = initial
        for index in sequence.indices {
            let phase = phaseSteps?[index] ?? 0
            precondition((0..<denominator).contains(phase))
            let continuesActivePlan = index > 0
                && sequence[index] == sequence[index - 1]
                && phase > (phaseSteps?[index - 1] ?? 0)
            if !continuesActivePlan {
                activePlanStart = terminal
            }
            starts.append(activePlanStart)
            terminal = continuedStructuredTrajectoryBoundary(
                sequence[index],
                progress: Float(phase + 1) / Float(denominator),
                initial: activePlanStart,
                armKnotCount: armKnotCount,
                blendKnotCount: blendKnotCount,
                legResidualKnotCount: legResidualKnotCount,
                maximumLegResidualAction: maximumLegResidualAction,
                torsoResidualKnotCount: torsoResidualKnotCount,
                maximumTorsoResidualAction: maximumTorsoResidualAction,
                armAsymmetryKnotCount: armAsymmetryKnotCount,
                maximumArmAsymmetryAction: maximumArmAsymmetryAction)
        }
        return StructuredTrajectorySequenceBoundaryReplay(
            starts: starts, terminal: terminal)
    }

    static func trajectoryHoldingSourceTerminalUpperBody(
        _ parameters: [Float],
        boundary: StructuredTrajectoryBoundary,
        armKnotCount: Int,
        blendKnotCount: Int,
        legResidualKnotCount: Int,
        torsoResidualKnotCount: Int,
        armAsymmetryKnotCount: Int,
        maximumArmAsymmetryAction: Float
    ) -> [Float] {
        precondition(armKnotCount > 0)
        precondition(boundary.armDelta.count == 8)
        precondition(boundary.armAsymmetry.count == 4)
        precondition(maximumArmAsymmetryAction > 0)
        var held = parameters
        let armParameterCount = 4 * armKnotCount
        let expectedCount = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount + torsoResidualKnotCount
            + 4 * armAsymmetryKnotCount
        precondition(held.count == expectedCount)
        for knot in 0..<armKnotCount {
            for action in 0..<4 {
                held[4 * knot + action] = boundary.armDelta[action]
            }
        }
        let asymmetryOffset = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount + torsoResidualKnotCount
        for knot in 0..<armAsymmetryKnotCount {
            for action in 0..<4 {
                held[asymmetryOffset + 4 * knot + action] = simd_clamp(
                    boundary.armAsymmetry[action]
                        / maximumArmAsymmetryAction,
                    -0.999, 0.999)
            }
        }
        return held
    }

    static func trajectoryTyingArmKnots(
        _ parameters: [Float],
        armKnotCount: Int,
        blendKnotCount: Int,
        legResidualKnotCount: Int,
        torsoResidualKnotCount: Int,
        armAsymmetryKnotCount: Int
    ) -> [Float] {
        precondition(armKnotCount > 0)
        var tied = parameters
        let armParameterCount = 4 * armKnotCount
        let expectedCount = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount + torsoResidualKnotCount
            + 4 * armAsymmetryKnotCount
        precondition(tied.count == expectedCount)
        let symmetric = Array(tied[0..<4])
        for knot in 1..<armKnotCount {
            for action in 0..<4 {
                tied[4 * knot + action] = symmetric[action]
            }
        }
        guard armAsymmetryKnotCount > 0 else { return tied }
        let asymmetryOffset = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount + torsoResidualKnotCount
        let asymmetry = Array(
            tied[asymmetryOffset..<(asymmetryOffset + 4)])
        for knot in 1..<armAsymmetryKnotCount {
            for action in 0..<4 {
                tied[asymmetryOffset + 4 * knot + action] =
                    asymmetry[action]
            }
        }
        return tied
    }

    static func trajectoryWithLocomotionProposal(
        _ parameters: [Float],
        blend: Float,
        armKnotCount: Int,
        blendKnotCount: Int,
        legResidualKnotCount: Int,
        torsoResidualKnotCount: Int,
        armAsymmetryKnotCount: Int,
        zeroResiduals: Bool
    ) -> [Float] {
        precondition(armKnotCount > 0 && blendKnotCount > 0)
        precondition(blend.isFinite && (0...1).contains(blend))
        var proposal = parameters
        let armParameterCount = 4 * armKnotCount
        let legOffset = armParameterCount + blendKnotCount
        let torsoOffset = legOffset + 10 * legResidualKnotCount
        let expectedCount = torsoOffset + torsoResidualKnotCount
            + 4 * armAsymmetryKnotCount
        precondition(proposal.count == expectedCount)
        for knot in 0..<blendKnotCount {
            proposal[armParameterCount + knot] = blend
        }
        if zeroResiduals {
            for index in legOffset..<(torsoOffset
                + torsoResidualKnotCount) {
                proposal[index] = 0
            }
        }
        return proposal
    }

    static func recedingEvaluationSteps(
        horizon: Int, phaseStep: Int, terminalHoldSteps: Int
    ) -> Int {
        precondition(horizon > 0)
        precondition((0..<horizon).contains(phaseStep))
        precondition(terminalHoldSteps >= 0)
        return horizon - phaseStep + terminalHoldSteps
    }

    static func recedingControlHorizonRemainder(
        configuredSteps: Int, horizon: Int, selectedPhaseStep: Int
    ) -> Int {
        precondition(configuredSteps > 0)
        precondition(horizon > 0)
        precondition((0..<horizon).contains(selectedPhaseStep))
        return min(
            configuredSteps - 1,
            horizon - selectedPhaseStep - 1)
    }

    static func recedingRequiredSafePrefixSteps(
        activePlanSteps: Int, controlHorizonSteps: Int,
        safetyLookaheadSteps: Int?
    ) -> Int {
        precondition(activePlanSteps > 0)
        precondition(controlHorizonSteps > 0)
        precondition(safetyLookaheadSteps.map { $0 >= 0 } ?? true)
        guard let safetyLookaheadSteps else {
            return activePlanSteps
        }
        return min(
            activePlanSteps,
            min(controlHorizonSteps, activePlanSteps)
                + safetyLookaheadSteps)
    }

    static func prioritizedValidationParameters(
        lossRanked: [[Float]], frontiers: [[Float]], limit: Int
    ) -> [[Float]] {
        precondition(limit > 0)
        var result = [[Float]]()
        result.reserveCapacity(limit)
        for parameters in frontiers + lossRanked {
            guard !result.contains(parameters) else { continue }
            result.append(parameters)
            if result.count == limit { break }
        }
        return result
    }

    static func isRecedingPlanEndpoint(
        predictedStep: Int, activePlanSteps: Int
    ) -> Bool {
        precondition(activePlanSteps > 0)
        precondition(predictedStep >= 0)
        return predictedStep == activePlanSteps - 1
    }

    static func recedingTerminalAbsoluteStep(
        committedStep: Int, horizon: Int, phaseStep: Int
    ) -> Int {
        precondition(committedStep >= 0)
        precondition(horizon > 0)
        precondition((0..<horizon).contains(phaseStep))
        return committedStep + horizon - phaseStep
    }

    static func pathCarryMinimum(
        terminalMinimum: Float, maximumRegression: Float
    ) -> Float {
        precondition(terminalMinimum.isFinite && terminalMinimum >= 0)
        precondition(maximumRegression.isFinite && maximumRegression >= 0)
        precondition(maximumRegression <= terminalMinimum)
        return terminalMinimum - maximumRegression
    }

    /// Project a whole-body carry observation onto the state actually owned
    /// by a leg-only proposal. The unloaded flat-walk actor was trained while
    /// controlling torso and arms; feeding it a loaded grasp posture and then
    /// discarding those outputs creates an out-of-distribution, non-causal
    /// dependency. Base motion, gravity, commands, legs, and leg action
    /// history remain measured; unowned torso/arm channels use the standing
    /// reference that defined the source checkpoint's zero coordinates.
    static func isolatedLegPolicyObservations(
        _ carryObservations: ContiguousArray<Float>,
        numEnvironments: Int
    ) -> ContiguousArray<Float> {
        let carryDimension = HumanoidBoxCarryTask.observationDimension
        let sourceDimension = HumanoidIsaacVelocityTask.observationDimension
        precondition(carryObservations.count
            == numEnvironments * carryDimension)
        var result = ContiguousArray<Float>()
        result.reserveCapacity(numEnvironments * sourceDimension)
        for environment in 0..<numEnvironments {
            let sourceBase = environment * carryDimension
            var row = ContiguousArray(
                carryObservations[
                    sourceBase..<(sourceBase + sourceDimension)])
            for joint in 10..<19 {
                row[12 + joint] = HumanoidWalkEnv
                    .defaultJointPositions[joint]
                row[31 + joint] = 0
                row[50 + joint] = 0
            }
            result.append(contentsOf: row)
        }
        return result
    }

    static func overrideLocomotionCommand(
        _ observations: inout ContiguousArray<Float>,
        numEnvironments: Int, forwardOnly: Bool, holonomic: Bool,
        speed commandedSpeed: Float?
    ) {
        precondition(!(forwardOnly && holonomic))
        precondition(observations.count
            == numEnvironments * HumanoidBoxCarryTask.observationDimension)
        guard forwardOnly || holonomic else { return }
        for environment in 0..<numEnvironments {
            let base = environment
                * HumanoidBoxCarryTask.observationDimension
            let currentX = observations[base + 9]
            let currentY = observations[base + 10]
            let speed = commandedSpeed ?? hypot(currentX, currentY)
            let goalX = observations[base + 103]
            let goalY = observations[base + 104]
            if holonomic {
                let norm = max(hypot(goalX, goalY), 1e-6)
                observations[base + 9] = speed * goalX / norm
                observations[base + 10] = speed * goalY / norm
                observations[base + 11] = 0
            } else {
                observations[base + 9] = speed
                observations[base + 10] = 0
                observations[base + 11] = simd_clamp(
                    0.5 * atan2(goalY, goalX), -1, 1)
            }
        }
    }

    private struct State {
        var humanoid: HumanoidState
        var manipulation: HumanoidManipulationState
    }

    private struct Flags {
        var bilateral: Bool
        var loadBearingGrasp: Bool
        var graspFrictionSupportFraction: Float
        var graspQuality: Float
        var unsupported: Bool
        var physicallyLifted: Bool
        var robotUpright: Bool
        var boxUpright: Bool
        var failed: Bool
    }

    private struct Candidate {
        var parameters: [Float]
        var metrics: HumanoidBoxPhysicalFlowMetrics
        var terminal: State
        var carryDistance: Float
        var warmupSteps: Int
    }

    private struct ValidatedCandidate {
        var candidate: Candidate
        var robustLoss: Float
        var successFraction: Float
    }

    private struct TargetDiscoveryCandidate {
        var parameters: [Float]
        var loss: Float
        var maximumNormalizedError: Float
        var bestStep: Int
        var trajectoryStart: StructuredTrajectoryBoundary? = nil
        var firstControlSafe: Bool = true
        var commitPathSafe: Bool = true
        var predictedPathSafe: Bool = true
        var terminalGoalFeasible: Bool = true
        var terminalGoalComponents: [String: Bool]? = nil
        var terminalRecoveryViable: Bool = true
        var jointValidationPassed: Bool = true
        var validationSuccessFraction: Float? = nil
        var phaseStep: Int = 0
        var maximumClearanceMeters: Float? = nil
        var maximumCarryDistanceMeters: Float? = nil
        var maximumStableCarryDistanceMeters: Float? = nil
        var maximumDestinationProgressMeters: Float? = nil
        var maximumRootDestinationProgressMeters: Float? = nil
        var maximumLoadedTouchdowns: Int? = nil
        var maximumLoadedAlternatingSteps: Int? = nil
        var maximumSwingFootLiftMeters: Float? = nil
        var maximumLoadedFootAirTimeSeconds: Float? = nil
        var maximumFootUnloadingFraction: Float? = nil
        var maximumGraspQuality: Float? = nil
        var maximumFeasibilityDwellSteps: Int? = nil
        var maximumPredicateDwellSteps: [Int]? = nil
        var firstStablePathViolationStep: Int? = nil
        var finalCarryDistanceMeters: Float? = nil
        var finalPlacementDistanceMeters: Float? = nil
        var finalDestinationProgressMeters: Float? = nil
        var finalRootDestinationProgressMeters: Float? = nil
        var finalLoadedTouchdowns: Int? = nil
        var finalLoadedAlternatingSteps: Int? = nil
        var finalClearanceMeters: Float? = nil
        var finalGraspQuality: Float? = nil
        var finalBoxVerticalVelocityMPS: Float? = nil
    }

    struct TargetDiscoverySwingFrontierScore: Equatable {
        var firstControlSafe: Bool
        var swingMilestonePassed: Bool
        var predictedPathSafe: Bool
        var terminalGoalFeasible: Bool
        var maximumFootAirTimeSeconds: Float
        var maximumSwingFootLiftMeters: Float
        var firstStablePathViolationStep: Int?
        var terminalRecoveryViable: Bool
        var terminalRecoveryMargin: Float
        var graspQualityDwellSteps: Int
        var loss: Float
    }

    struct TargetDiscoveryFeasibilityFrontierScore: Equatable {
        var firstControlSafe: Bool
        var predictedPathSafe: Bool
        var maximumFeasibilityDwellSteps: Int
        var terminalGoalFeasible: Bool
        var terminalRecoveryViable: Bool
        var terminalRecoveryMargin: Float
        var loss: Float
    }

    struct TargetDiscoveryPathScore: Equatable {
        var firstControlSafe: Bool
        var commitPathSafe: Bool
        var terminalGoalFeasible: Bool
        var predictedPathSafe: Bool
        var firstStablePathViolationStep: Int?
        var maximumFeasibilityDwellSteps: Int
        var terminalRecoveryViable: Bool
        var loss: Float
    }

    /// Order CEM candidates by the same non-tradeable contract used for
    /// committing a physical control. Scalar endpoint losses remain the final
    /// tie-breaker, but cannot make an earlier unstable fall outrank a later
    /// path-safe prefix merely because the terminal state looks better.
    static func pathScoreIsBetter(
        _ candidate: TargetDiscoveryPathScore,
        than incumbent: TargetDiscoveryPathScore
    ) -> Bool {
        if candidate.firstControlSafe != incumbent.firstControlSafe {
            return candidate.firstControlSafe
        }
        if candidate.commitPathSafe != incumbent.commitPathSafe {
            return candidate.commitPathSafe
        }
        if candidate.predictedPathSafe != incumbent.predictedPathSafe {
            return candidate.predictedPathSafe
        }
        // Once both candidates are known to violate the predicted path,
        // preserve the prefix that remains valid longer. A terminal snapshot
        // cannot redeem an earlier intervening failure.
        if !candidate.predictedPathSafe {
            let candidateViolation =
                candidate.firstStablePathViolationStep ?? Int.max
            let incumbentViolation =
                incumbent.firstStablePathViolationStep ?? Int.max
            if candidateViolation != incumbentViolation {
                return candidateViolation > incumbentViolation
            }
        }
        if candidate.terminalRecoveryViable
                != incumbent.terminalRecoveryViable {
            return candidate.terminalRecoveryViable
        }
        if candidate.terminalGoalFeasible
                != incumbent.terminalGoalFeasible {
            return candidate.terminalGoalFeasible
        }
        if candidate.maximumFeasibilityDwellSteps
                != incumbent.maximumFeasibilityDwellSteps {
            return candidate.maximumFeasibilityDwellSteps
                > incumbent.maximumFeasibilityDwellSteps
        }
        return candidate.loss < incumbent.loss
    }

    static func feasibilityFrontierIsBetter(
        _ candidate: TargetDiscoveryFeasibilityFrontierScore,
        than incumbent: TargetDiscoveryFeasibilityFrontierScore?
    ) -> Bool {
        guard let incumbent else { return true }
        if candidate.firstControlSafe != incumbent.firstControlSafe {
            return candidate.firstControlSafe
        }
        if candidate.predictedPathSafe != incumbent.predictedPathSafe {
            return candidate.predictedPathSafe
        }
        if candidate.maximumFeasibilityDwellSteps
                != incumbent.maximumFeasibilityDwellSteps {
            return candidate.maximumFeasibilityDwellSteps
                > incumbent.maximumFeasibilityDwellSteps
        }
        if candidate.terminalGoalFeasible
                != incumbent.terminalGoalFeasible {
            return candidate.terminalGoalFeasible
        }
        if candidate.terminalRecoveryViable
                != incumbent.terminalRecoveryViable {
            return candidate.terminalRecoveryViable
        }
        if candidate.terminalRecoveryMargin
                != incumbent.terminalRecoveryMargin {
            return candidate.terminalRecoveryMargin
                > incumbent.terminalRecoveryMargin
        }
        return candidate.loss < incumbent.loss
    }

    static func swingFrontierIsBetter(
        _ candidate: TargetDiscoverySwingFrontierScore,
        than incumbent: TargetDiscoverySwingFrontierScore?
    ) -> Bool {
        guard let incumbent else { return true }
        if candidate.firstControlSafe != incumbent.firstControlSafe {
            return candidate.firstControlSafe
        }
        if candidate.swingMilestonePassed
                != incumbent.swingMilestonePassed {
            return candidate.swingMilestonePassed
        }
        if candidate.predictedPathSafe != incumbent.predictedPathSafe {
            return candidate.predictedPathSafe
        }
        if candidate.terminalGoalFeasible
                != incumbent.terminalGoalFeasible {
            return candidate.terminalGoalFeasible
        }
        let candidateViolation =
            candidate.firstStablePathViolationStep ?? Int.max
        let incumbentViolation =
            incumbent.firstStablePathViolationStep ?? Int.max
        if candidateViolation != incumbentViolation {
            return candidateViolation > incumbentViolation
        }
        if candidate.terminalRecoveryViable
                != incumbent.terminalRecoveryViable {
            return candidate.terminalRecoveryViable
        }
        if candidate.terminalRecoveryMargin
                != incumbent.terminalRecoveryMargin {
            return candidate.terminalRecoveryMargin
                > incumbent.terminalRecoveryMargin
        }
        if candidate.graspQualityDwellSteps
                != incumbent.graspQualityDwellSteps {
            return candidate.graspQualityDwellSteps
                > incumbent.graspQualityDwellSteps
        }
        if candidate.maximumFootAirTimeSeconds
                != incumbent.maximumFootAirTimeSeconds {
            return candidate.maximumFootAirTimeSeconds
                > incumbent.maximumFootAirTimeSeconds
        }
        if candidate.maximumSwingFootLiftMeters
                != incumbent.maximumSwingFootLiftMeters {
            return candidate.maximumSwingFootLiftMeters
                > incumbent.maximumSwingFootLiftMeters
        }
        return candidate.loss < incumbent.loss
    }

    private enum TargetDiscoverySearchBlock {
        case configured
        case lowerBody
        case upperBody
        case joint
    }

    private struct Target {
        var state: State
        var flags: Flags
        var step: Int
        var clearance: Float
        var boxUpright: Float
        var robotUpright: Float
        var carryDistance: Float
        var placementDistance: Float
        var destinationProgress: Float
        var rootDestinationProgress: Float
        var loadedTouchdowns: Int
        var loadedAlternatingSteps: Int
        var maximumSwingFootLift: Float
        var maximumLoadedFootAirTime: Float = 0
        var maximumFootUnloadingFraction: Float = 0
        var terminalFootUnloadingFraction: Float = 0
        var replicas: [State]
        var replicaFlags: [Flags]
        var replicaCarryDistances: [Float]
        var replicaClearances: [Float]
        var replicaGraspQualities: [Float]
        var replicaFootUnloadingFractions: [Float]
        var warmupSteps: Int
        var simulatedPreparationSteps: Int
        var sourceReplaySuccessFraction: Float
        var stableCarryPath: Bool
        var predictedRecoveryPathSafe: Bool = true
        var generatingTrajectorySequence: [[Float]]? = nil
        var generatingTrajectorySequencePhaseSteps: [Int]? = nil
        var committedTrace: [HumanoidBoxPhysicalFlowTraceSample]? = nil
        var graspFeedback: ObjectRelativeGraspFeedback? = nil
    }

    private struct WarmTask {
        var task: HumanoidBoxCarryTask
        var observation: RLObservationBatch
        var result: RLStepBatch
        var warmupSteps: Int
        var simulatedPreparationSteps: Int
        var sourceReplaySuccessFraction: Float
    }

    private struct CachedSource {
        var runtime: WarmTask
        var snapshot: HumanoidBoxCarryTask.SpeculationSnapshot
    }

    public static func run(
        checkpointDirectory: String,
        targetTrajectory: [Float],
        proposalTrajectory: [Float]?,
        archivedTargetTrajectorySequence: [[Float]]? = nil,
        archivedTargetTrajectorySequencePhaseSteps: [Int]? = nil,
        archivedTargetPredictedRecoveryPathSafe: Bool? = nil,
        archivedSourceWarmupActions: [[Float]]? = nil,
        sourceStages: [HumanoidBoxPhysicalFlowStage] = [],
        configuration: HumanoidBoxPhysicalFlowConfiguration = .init()
    ) throws -> HumanoidBoxPhysicalFlowReport {
        try configuration.validate()
        let armParameterCount = 4 * configuration.trajectoryKnotCount
        let legBlendParameterCount = configuration.legBlendKnotCount
        let legResidualParameterCount = 10
            * configuration.legResidualKnotCount
        let torsoResidualParameterCount =
            configuration.torsoResidualKnotCount
        let armAsymmetryParameterCount = 4
            * configuration.armAsymmetryKnotCount
        let parameterCount = armParameterCount
            + legBlendParameterCount + legResidualParameterCount
            + torsoResidualParameterCount + armAsymmetryParameterCount
        let armAsymmetryOffset = armParameterCount + legBlendParameterCount
            + legResidualParameterCount + torsoResidualParameterCount

        func targetDiscoveryParameterIsActive(
            _ index: Int,
            block: TargetDiscoverySearchBlock = .configured
        ) -> Bool {
            switch block {
            case .lowerBody:
                return index >= armParameterCount
                    && index < armAsymmetryOffset
            case .upperBody:
                return index < armParameterCount
                    || index >= armAsymmetryOffset
            case .joint:
                return true
            case .configured:
                break
            }
            if configuration.targetDiscoveryArmOnly {
                return index < armParameterCount
                    || index >= armAsymmetryOffset
            }
            if configuration.targetDiscoveryLowerBodyOnly {
                return index >= armParameterCount
                    && index < armAsymmetryOffset
            }
            return true
        }

        func projectedTargetDiscoveryTrajectory(
            _ parameters: [Float],
            onto center: [Float],
            block: TargetDiscoverySearchBlock
        ) -> [Float] {
            precondition(parameters.count == center.count)
            return parameters.indices.map {
                targetDiscoveryParameterIsActive($0, block: block)
                    ? parameters[$0] : center[$0]
            }
        }

        func structuredTargetDiscoveryTrajectory(
            _ parameters: [Float]
        ) -> [Float] {
            guard configuration.targetDiscoveryTiedArmKnots else {
                return parameters
            }
            return trajectoryTyingArmKnots(
                parameters,
                armKnotCount: configuration.trajectoryKnotCount,
                blendKnotCount: configuration.legBlendKnotCount,
                legResidualKnotCount:
                    configuration.legResidualKnotCount,
                torsoResidualKnotCount:
                    configuration.torsoResidualKnotCount,
                armAsymmetryKnotCount:
                    configuration.armAsymmetryKnotCount)
        }

        func normalizedTrajectory(_ values: [Float]) -> [Float]? {
            guard values.allSatisfy(\.isFinite) else { return nil }
            if values.count == parameterCount { return values }
            if values.count == armParameterCount {
                return values + [Float](
                    repeating: 0,
                    count: legBlendParameterCount
                        + legResidualParameterCount
                        + torsoResidualParameterCount
                        + armAsymmetryParameterCount)
            }
            if values.count == armParameterCount + legBlendParameterCount,
               legResidualParameterCount + torsoResidualParameterCount
                    + armAsymmetryParameterCount > 0 {
                return values + [Float](
                    repeating: 0, count: legResidualParameterCount
                        + torsoResidualParameterCount
                        + armAsymmetryParameterCount)
            }
            if values.count == armParameterCount + legBlendParameterCount
                    + legResidualParameterCount,
               torsoResidualParameterCount + armAsymmetryParameterCount > 0 {
                return values + [Float](
                    repeating: 0, count: torsoResidualParameterCount
                        + armAsymmetryParameterCount)
            }
            if values.count == armParameterCount + legBlendParameterCount
                    + legResidualParameterCount
                    + torsoResidualParameterCount,
               armAsymmetryParameterCount > 0 {
                return values + [Float](
                    repeating: 0, count: armAsymmetryParameterCount)
            }
            return nil
        }
        let archivedTargetTrajectoryCount =
            archivedTargetTrajectorySequence?.count
        let archivedTargetTrajectorySequence =
            archivedTargetTrajectorySequence?.compactMap(
                normalizedTrajectory)
        guard archivedTargetTrajectorySequence?.count
                == archivedTargetTrajectoryCount,
              (archivedTargetTrajectorySequencePhaseSteps == nil
                || archivedTargetTrajectorySequencePhaseSteps?.count
                    == archivedTargetTrajectorySequence?.count),
              (archivedTargetTrajectorySequencePhaseSteps?.allSatisfy {
                  $0 >= 0
                    && $0 < configuration.recedingHorizonSteps
              } ?? true) else {
            throw RLEnvironmentError.invalidConfiguration(
                "archived target sequence does not match the trajectory schema")
        }

        func validGraspFeedback(
            _ stage: HumanoidBoxPhysicalFlowStage
        ) -> Bool {
            let values = [
                stage.leftGraspAnchorBoxLocalMeters,
                stage.rightGraspAnchorBoxLocalMeters,
            ]
            if let blend = stage.graspAnchorFeedbackBlend {
                return blend.isFinite && blend > 0 && blend <= 1
                    && stage.graspAnchorFeedbackVelocityHorizonSeconds.map {
                        $0.isFinite && (0...0.2).contains($0)
                    } == true
                    && stage
                        .graspAnchorFeedbackMaximumActionCorrection.map {
                            $0.isFinite && $0 > 0 && $0 <= 1
                        } == true
                    && stage.graspAnchorFeedbackInwardPreloadMeters.map {
                        $0.isFinite && (0...0.03).contains($0)
                    } == true
                    && stage.graspAnchorBoxHeightMeters?.isFinite == true
                    && values.allSatisfy {
                        $0?.count == 3
                            && $0!.allSatisfy(\.isFinite)
                    }
            }
            return values.allSatisfy { $0 == nil }
                && stage.graspAnchorFeedbackVelocityHorizonSeconds == nil
                && stage.graspAnchorFeedbackMaximumActionCorrection == nil
                && stage.graspAnchorFeedbackInwardPreloadMeters == nil
                && stage.graspAnchorBoxHeightMeters == nil
        }

        guard let targetTrajectory = normalizedTrajectory(targetTrajectory),
              proposalTrajectory == nil
                || normalizedTrajectory(proposalTrajectory!) != nil,
              sourceStages.allSatisfy({ stage in
                  stage.controlSteps > 0
                      && (stage.appliedNormalizedActions.map {
                          $0.count == stage.controlSteps
                              && $0.allSatisfy {
                                  $0.count
                                      == firstArmAction + armActionCount
                                      && $0.allSatisfy(\.isFinite)
                              }
                          } ?? true)
                      && validGraspFeedback(stage)
                      && (stage.minimumCarryDistanceMeters.map {
                          $0.isFinite && $0 >= 0
                      } ?? true)
                      && (stage.minimumDestinationProgressMeters.map {
                          $0.isFinite && $0 >= 0
                      } ?? true)
                      && (stage.minimumFootAirTimeSeconds.map {
                          $0.isFinite && $0 >= 0 && $0 <= 1
                      } ?? true)
                      && (stage.minimumSwingFootLiftMeters.map {
                          $0.isFinite && $0 >= 0 && $0 <= 0.25
                      } ?? true)
                      && (stage.minimumFootUnloadingFraction.map {
                          $0.isFinite && (0...1).contains($0)
                      } ?? true)
                      && (stage.minimumTerminalFootUnloadingFraction.map {
                          $0.isFinite && (0...1).contains($0)
                      } ?? true)
                      && (stage.minimumClearanceMeters.map {
                          $0.isFinite && $0 >= 0.01
                      } ?? true)
                      && (stage.maximumPathDownwardBoxVelocityMPS.map {
                          $0.isFinite && $0 >= 0
                      } ?? true)
                      && (stage.minimumGraspQuality.map {
                          $0.isFinite && (0...1).contains($0)
                      } ?? true)
                      && (stage.certificationDwellSteps.map { $0 > 0 }
                        ?? true)
                      && (stage.locomotionCommandSpeed.map {
                          $0.isFinite && $0 > 0 && $0 <= 1
                            && stage.locomotionCheckpointDirectory != nil
                            && ((stage.forwardOnlyBaseCommand == true)
                                != (stage.holonomicBaseCommand == true))
                      } ?? true)
                      && !(stage.forwardOnlyBaseCommand == true
                        && stage.holonomicBaseCommand == true)
                      && (stage.policyOnly == true
                        ? stage.trajectory.isEmpty
                            && stage.trajectoryDurationSteps == nil
                            && stage.trajectorySequence == nil
                            && stage.trajectorySequencePhaseSteps == nil
                            && stage.locomotionCheckpointDirectory == nil
                            && stage.locomotionCommandSpeed == nil
                            && stage.graspAnchorFeedbackBlend == nil
                            && stage
                                .graspAnchorFeedbackVelocityHorizonSeconds
                                    == nil
                            && stage
                                .graspAnchorFeedbackMaximumActionCorrection
                                    == nil
                            && stage
                                .graspAnchorFeedbackInwardPreloadMeters
                                    == nil
                            && stage.leftGraspAnchorBoxLocalMeters == nil
                            && stage.rightGraspAnchorBoxLocalMeters == nil
                            && stage.graspAnchorBoxHeightMeters == nil
                        : (stage.trajectorySequence.map { sequence in
                            sequence.count == stage.controlSteps
                                && (stage.trajectorySequenceStepDenominator
                                    ?? 0) > 0
                                && (stage.trajectorySequencePhaseSteps.map {
                                    $0.count == stage.controlSteps
                                        && $0.allSatisfy {
                                            $0 >= 0 && $0 < stage
                                                .trajectorySequenceStepDenominator!
                                        }
                                } ?? true)
                                && sequence.allSatisfy {
                                    normalizedTrajectory($0) != nil
                                }
                          } ?? ((stage.trajectoryDurationSteps
                                    ?? stage.controlSteps) > 0
                                && normalizedTrajectory(stage.trajectory)
                                    != nil)))
              }),
              archivedSourceWarmupActions.map({
                  !$0.isEmpty
                      && $0.allSatisfy {
                          $0.count == firstArmAction + armActionCount
                              && $0.allSatisfy(\.isFinite)
                      }
              }) ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "humanoid-box flow trajectories do not match the knot schema")
        }
        let proposalTrajectory = proposalTrajectory.flatMap(
            normalizedTrajectory)
        let sourceStages = sourceStages.map {
            if $0.policyOnly == true {
                return HumanoidBoxPhysicalFlowStage(
                    trajectory: [], controlSteps: $0.controlSteps,
                    policyOnly: true,
                    appliedNormalizedActions:
                        $0.appliedNormalizedActions,
                    canonicalizeReplicasBeforeExecution:
                        $0.canonicalizeReplicasBeforeExecution == true,
                    continueFromPreviousTrajectoryTerminal:
                        $0.continueFromPreviousTrajectoryTerminal == true,
                    graspAnchorFeedbackBlend:
                        $0.graspAnchorFeedbackBlend,
                    graspAnchorFeedbackVelocityHorizonSeconds:
                        $0.graspAnchorFeedbackVelocityHorizonSeconds,
                    graspAnchorFeedbackMaximumActionCorrection:
                        $0.graspAnchorFeedbackMaximumActionCorrection,
                    graspAnchorFeedbackInwardPreloadMeters:
                        $0.graspAnchorFeedbackInwardPreloadMeters,
                    leftGraspAnchorBoxLocalMeters:
                        $0.leftGraspAnchorBoxLocalMeters,
                    rightGraspAnchorBoxLocalMeters:
                        $0.rightGraspAnchorBoxLocalMeters,
                    graspAnchorBoxHeightMeters:
                        $0.graspAnchorBoxHeightMeters,
                    minimumCarryDistanceMeters:
                        $0.minimumCarryDistanceMeters,
                    minimumDestinationProgressMeters:
                        $0.minimumDestinationProgressMeters,
                    minimumRootDestinationProgressMeters:
                        $0.minimumRootDestinationProgressMeters,
                    minimumTouchdowns:
                        $0.minimumTouchdowns,
                    minimumAlternatingSteps:
                        $0.minimumAlternatingSteps,
                    minimumSwingFootLiftMeters:
                        $0.minimumSwingFootLiftMeters,
                    minimumFootAirTimeSeconds:
                        $0.minimumFootAirTimeSeconds,
                    minimumFootUnloadingFraction:
                        $0.minimumFootUnloadingFraction,
                    minimumTerminalFootUnloadingFraction:
                        $0.minimumTerminalFootUnloadingFraction,
                    minimumClearanceMeters:
                        $0.minimumClearanceMeters,
                    maximumPathDownwardBoxVelocityMPS:
                        $0.maximumPathDownwardBoxVelocityMPS,
                    minimumGraspQuality:
                        $0.minimumGraspQuality,
                    certificationDwellSteps:
                        $0.certificationDwellSteps)
            }
            return HumanoidBoxPhysicalFlowStage(
                trajectory: $0.trajectorySequence == nil
                    ? normalizedTrajectory($0.trajectory)! : [],
                controlSteps: $0.controlSteps,
                trajectoryDurationSteps: $0.trajectoryDurationSteps,
                trajectorySequence: $0.trajectorySequence?.map {
                    normalizedTrajectory($0)!
                },
                trajectorySequencePhaseSteps:
                    $0.trajectorySequencePhaseSteps,
                trajectorySequenceStepDenominator:
                    $0.trajectorySequenceStepDenominator,
                forwardOnlyBaseCommand:
                    $0.forwardOnlyBaseCommand == true,
                holonomicBaseCommand:
                    $0.holonomicBaseCommand == true,
                locomotionCheckpointDirectory:
                    $0.locomotionCheckpointDirectory,
                locomotionCommandSpeed: $0.locomotionCommandSpeed,
                policyOnly: false,
                appliedNormalizedActions:
                    $0.appliedNormalizedActions,
                canonicalizeReplicasBeforeExecution:
                    $0.canonicalizeReplicasBeforeExecution == true,
                continueFromPreviousTrajectoryTerminal:
                    $0.continueFromPreviousTrajectoryTerminal == true,
                graspAnchorFeedbackBlend:
                    $0.graspAnchorFeedbackBlend,
                graspAnchorFeedbackVelocityHorizonSeconds:
                    $0.graspAnchorFeedbackVelocityHorizonSeconds,
                graspAnchorFeedbackMaximumActionCorrection:
                    $0.graspAnchorFeedbackMaximumActionCorrection,
                graspAnchorFeedbackInwardPreloadMeters:
                    $0.graspAnchorFeedbackInwardPreloadMeters,
                leftGraspAnchorBoxLocalMeters:
                    $0.leftGraspAnchorBoxLocalMeters,
                rightGraspAnchorBoxLocalMeters:
                    $0.rightGraspAnchorBoxLocalMeters,
                graspAnchorBoxHeightMeters:
                    $0.graspAnchorBoxHeightMeters,
                minimumCarryDistanceMeters:
                    $0.minimumCarryDistanceMeters,
                minimumDestinationProgressMeters:
                    $0.minimumDestinationProgressMeters,
                minimumRootDestinationProgressMeters:
                    $0.minimumRootDestinationProgressMeters,
                minimumTouchdowns:
                    $0.minimumTouchdowns,
                minimumAlternatingSteps:
                    $0.minimumAlternatingSteps,
                minimumSwingFootLiftMeters:
                    $0.minimumSwingFootLiftMeters,
                minimumFootAirTimeSeconds:
                    $0.minimumFootAirTimeSeconds,
                minimumFootUnloadingFraction:
                    $0.minimumFootUnloadingFraction,
                minimumTerminalFootUnloadingFraction:
                    $0.minimumTerminalFootUnloadingFraction,
                minimumClearanceMeters:
                    $0.minimumClearanceMeters,
                maximumPathDownwardBoxVelocityMPS:
                    $0.maximumPathDownwardBoxVelocityMPS,
                minimumGraspQuality:
                    $0.minimumGraspQuality,
                certificationDwellSteps:
                    $0.certificationDwellSteps)
        }
        if let proposalTrajectory, proposalTrajectory == targetTrajectory {
            throw RLEnvironmentError.invalidConfiguration(
                "target-generating trajectory must be withheld from search")
        }
        func terminalBoundary(
            for stage: HumanoidBoxPhysicalFlowStage
        ) -> StructuredTrajectoryBoundary? {
            guard stage.policyOnly != true, stage.controlSteps > 0 else {
                return nil
            }
            let finalControl = stage.controlSteps - 1
            let parameters = stage.trajectorySequence?[finalControl]
                ?? stage.trajectory
            let denominator = stage.trajectorySequence == nil
                ? (stage.trajectoryDurationSteps ?? stage.controlSteps)
                : stage.trajectorySequenceStepDenominator!
            let progress = Float(
                stage.trajectoryEvaluationStep(at: finalControl) + 1)
                / Float(denominator)
            return structuredTrajectoryBoundary(
                parameters,
                progress: progress,
                armKnotCount: configuration.trajectoryKnotCount,
                blendKnotCount: configuration.legBlendKnotCount,
                legResidualKnotCount:
                    configuration.legResidualKnotCount,
                maximumLegResidualAction:
                    configuration.maximumLegResidualAction,
                torsoResidualKnotCount:
                    configuration.torsoResidualKnotCount,
                maximumTorsoResidualAction:
                    configuration.maximumTorsoResidualAction,
                armAsymmetryKnotCount:
                    configuration.armAsymmetryKnotCount,
                maximumArmAsymmetryAction:
                    configuration.maximumArmAsymmetryAction)
        }
        let sourceTerminalBoundary = sourceStages.reversed()
            .compactMap { terminalBoundary(for: $0) }.first
        if configuration.continueTrajectoryFromSourceTerminal,
           sourceTerminalBoundary == nil {
            throw RLEnvironmentError.invalidConfiguration(
                "trajectory continuation requires a preceding physical-flow stage")
        }
        let currentTrajectoryStart =
            configuration.continueTrajectoryFromSourceTerminal
                ? sourceTerminalBoundary : nil
        let sourceControlSteps = sourceStages.reduce(0) {
            $0 + $1.controlSteps
        }
        let targetExecutionSteps = configuration.targetExecutionSteps
            ?? configuration.targetGenerationSteps
        guard archivedTargetTrajectorySequence?.count
                == targetExecutionSteps
                || archivedTargetTrajectorySequence == nil,
              archivedTargetTrajectorySequence == nil
                || configuration.recedingHorizonSteps > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "archived target sequence must cover the exact receding target")
        }
        let minimumTargetPathCarryDistance = pathCarryMinimum(
            terminalMinimum:
                configuration.minimumTargetCarryDistanceMeters,
            maximumRegression:
                configuration.maximumTargetPathCarryRegressionMeters)

        let startTime = Date()
        let optimizerSeed = configuration.optimizerSeed
            ?? configuration.seed
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        guard runner.metadata.task == "humanoid-box-carry-v0",
              let semanticOptions = runner.metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "humanoid-box flow requires a configured box-carry checkpoint")
        }
        var replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: runner.metadata.task,
            semanticOptions: semanticOptions,
            maxEpisodeSteps: runner.metadata.maxEpisodeSteps,
            controlDecimation: runner.metadata.controlDecimation)
        if let fraction = configuration
                .carryBaseLegActionFractionOverride {
            replayOptions["carryBaseLegActionFraction"] = fraction
        }
        let locomotionDirectories = Set(
            sourceStages.compactMap(\.locomotionCheckpointDirectory)
                + [configuration.recedingLocomotionCheckpointDirectory]
                    .compactMap { $0 })
        var locomotionRunners = [String: VectorPolicyRunner]()
        for directory in locomotionDirectories.sorted() {
            let locomotionRunner = try VectorPolicyRunner(
                checkpointDirectory: directory)
            guard locomotionRunner.metadata.task == "humanoid-isaac-flat-v0",
                  locomotionRunner.metadata.observationDimension
                    == HumanoidIsaacVelocityTask.observationDimension,
                  locomotionRunner.metadata.actionDimension
                    == runner.metadata.actionDimension else {
                throw RLEnvironmentError.invalidConfiguration(
                    "locomotion proposal must be a 69x19 Isaac H1 flat-walk checkpoint")
            }
            locomotionRunners[directory] = locomotionRunner
        }

        func policyActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch
        ) throws -> RLActionBatch {
            return try runner.actions(
                for: observation,
                expertGates: task.policyExpertGates(observation.policy),
                expertActionMask: task.policyExpertActionMask,
                standExpertGates:
                    task.policyStandExpertGates(observation.policy),
                standExpertActionMask: task.policyStandExpertActionMask,
                auxiliaryExpertGates:
                    task.policyAuxiliaryExpertGates(observation.policy),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }

        func baseLegPolicyActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            forwardOnlyCommand: Bool,
            holonomicCommand: Bool,
            locomotionCheckpointDirectory: String?,
            locomotionCommandSpeed: Float?
        ) throws -> RLActionBatch {
            var baseObservation = observation
            overrideLocomotionCommand(
                &baseObservation.policy,
                numEnvironments: task.spec.numEnvironments,
                forwardOnly: forwardOnlyCommand,
                holonomic: holonomicCommand,
                speed: locomotionCommandSpeed)
            if let locomotionCheckpointDirectory {
                guard let locomotionRunner = locomotionRunners[
                        locomotionCheckpointDirectory] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "locomotion proposal checkpoint was not loaded")
                }
                let sourceObservations = isolatedLegPolicyObservations(
                    baseObservation.policy,
                    numEnvironments: task.spec.numEnvironments)
                let values = try locomotionRunner.actions(
                    for: sourceObservations)
                return try RLActionBatch(
                    numEnvironments: task.spec.numEnvironments,
                    actionDimension: task.spec.action.elementCount,
                    values: values)
            }
            return try runner.actions(
                for: baseObservation,
                expertGates: task.policyExpertGates(baseObservation.policy),
                expertActionMask: task.policyExpertActionMask,
                standExpertGates:
                    task.policyStandExpertGates(baseObservation.policy),
                standExpertActionMask: task.policyStandExpertActionMask,
                auxiliaryExpertGates: ContiguousArray(
                    repeating: 0, count: task.spec.numEnvironments),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }

        var sourceWarmupAppliedActions: [[Float]]?

        func makeWarmTask(count: Int) throws -> WarmTask {
            let anyTask = try BuiltInRLTasks.registry.make(
                runner.metadata.task,
                configuration: RLTaskConfiguration(
                    numEnvironments: count,
                    seed: configuration.seed,
                    autoReset: false,
                    options: replayOptions))
            guard let task = anyTask as? HumanoidBoxCarryTask else {
                throw RLEnvironmentError.invalidConfiguration(
                    "registered box-carry task has an unexpected implementation")
            }
            var compatibilityMetadata = runner.metadata
            if let fraction = configuration
                    .carryBaseLegActionFractionOverride {
                compatibilityMetadata.taskConfiguration?[
                    "carryBaseLegActionFraction"] = fraction
            }
            let compatibility = compatibilityMetadata
                .compatibilityMismatches(with: task.spec)
            guard compatibility.isEmpty else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-box flow checkpoint/task mismatch: "
                        + compatibility.joined(separator: "; "))
            }
            var observation = try task.reset(seed: configuration.seed)
            var result = RLStepBatch(spec: task.spec)
            var contactStreak = 0
            var warmupSteps = 0
            var executedWarmupActions = [[Float]]()
            if let archivedSourceWarmupActions {
                for committed in archivedSourceWarmupActions {
                    var values = ContiguousArray<Float>()
                    values.reserveCapacity(
                        count * task.spec.action.elementCount)
                    for _ in 0..<count {
                        values.append(contentsOf: committed)
                    }
                    let actions = try RLActionBatch(
                        numEnvironments: count,
                        actionDimension: task.spec.action.elementCount,
                        values: values)
                    executedWarmupActions.append(committed)
                    try task.step(actions: actions, into: &result)
                    warmupSteps += 1
                    observation = result.observations
                }
                let left = result.metrics["state/left_hand_contact"]!
                let right = result.metrics["state/right_hand_contact"]!
                let allBilateral = zip(left, right).allSatisfy {
                    $0.0 > 0.5 && $0.1 > 0.5
                }
                contactStreak = allBilateral
                    ? configuration.contactDwellSteps : 0
            } else {
                while warmupSteps < configuration.maximumWarmupSteps,
                      contactStreak < configuration.contactDwellSteps {
                    let actions = try policyActions(
                        task: task, observation: observation)
                    executedWarmupActions.append(Array(
                        actions.values.prefix(
                            task.spec.action.elementCount)))
                    try task.step(actions: actions, into: &result)
                    warmupSteps += 1
                    observation = result.observations
                    let left = result.metrics["state/left_hand_contact"]!
                    let right = result.metrics["state/right_hand_contact"]!
                    let allBilateral = zip(left, right).allSatisfy {
                        $0.0 > 0.5 && $0.1 > 0.5
                    }
                    contactStreak = allBilateral ? contactStreak + 1 : 0
                    if result.terminated.contains(true)
                        || result.truncated.contains(true) {
                        break
                    }
                }
            }
            guard contactStreak >= configuration.contactDwellSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "checkpoint failed to establish the physical-flow source grasp")
            }
            if sourceWarmupAppliedActions == nil {
                sourceWarmupAppliedActions = executedWarmupActions
            }
            return WarmTask(
                task: task, observation: observation, result: result,
                warmupSteps: warmupSteps,
                simulatedPreparationSteps: warmupSteps,
                sourceReplaySuccessFraction: 1)
        }

        func captureGraspFeedback(
            task: HumanoidBoxCarryTask, blend: Float
        ) -> ObjectRelativeGraspFeedback {
            precondition(blend > 0 && blend <= 1)
            let state = task.environment.manipulationStates()[0]
            let inverseBox = state.object.rotation.conjugate
            return ObjectRelativeGraspFeedback(
                blend: blend,
                velocityHorizonSeconds:
                    configuration
                        .graspAnchorFeedbackVelocityHorizonSeconds,
                maximumActionCorrection:
                    configuration
                        .graspAnchorFeedbackMaximumActionCorrection,
                inwardPreloadMeters:
                    configuration
                        .graspAnchorFeedbackInwardPreloadMeters,
                leftAnchorBoxLocal: inverseBox.act(
                    state.leftHand.position - state.object.position),
                rightAnchorBoxLocal: inverseBox.act(
                    state.rightHand.position - state.object.position),
                boxHeightMeters: state.object.position.z)
        }

        func graspFeedbackArmDiagnostics(
            task: HumanoidBoxCarryTask,
            feedback: ObjectRelativeGraspFeedback
        ) -> [[GraspFeedbackArmDiagnostic]] {
            let manipulation = task.environment.manipulationStates()
            let bodyIndices = task.environment.refs.flatMap { reference in
                (firstArmAction..<(firstArmAction + armActionCount)).map {
                    action in
                    task.environment.scene.joints[
                        reference.motors[action]].bodyB
                }
            }
            let jointBodies = task.environment.solver.bodyStates(bodyIndices)
            return (0..<task.spec.numEnvironments).map { environment in
                let reference = task.environment.refs[environment]
                let object = manipulation[environment].object
                var diagnostics = [GraspFeedbackArmDiagnostic]()
                diagnostics.reserveCapacity(2)
                for side in 0..<2 {
                    let firstJoint = firstArmAction + 4 * side
                    let hand = side == 0
                        ? manipulation[environment].leftHand
                        : manipulation[environment].rightHand
                    let localAnchor = side == 0
                        ? feedback.leftAnchorBoxLocal
                        : feedback.rightAnchorBoxLocal
                    var preloadedAnchor = localAnchor
                    if preloadedAnchor.y > 0 {
                        preloadedAnchor.y -= feedback.inwardPreloadMeters
                    } else if preloadedAnchor.y < 0 {
                        preloadedAnchor.y += feedback.inwardPreloadMeters
                    }
                    let worldOffset = object.rotation.act(preloadedAnchor)
                    var desiredPosition = object.position + worldOffset
                    desiredPosition.z += feedback.boxHeightMeters
                        - object.position.z
                    var desiredVelocity = object.linearVelocity
                        + cross(object.angularVelocity, worldOffset)
                    desiredVelocity.z = cross(
                        object.angularVelocity, worldOffset).z
                    var taskDelta = desiredPosition - hand.position
                        + feedback.velocityHorizonSeconds
                            * (desiredVelocity - hand.linearVelocity)
                    let unclampedTaskMagnitude = length(taskDelta)
                    if unclampedTaskMagnitude > 0.08 {
                        taskDelta *= 0.08 / unclampedTaskMagnitude
                    }
                    var jacobian = [F3]()
                    jacobian.reserveCapacity(4)
                    for jointOffset in 0..<4 {
                        let action = firstJoint + jointOffset
                        let joint = task.environment.scene.joints[
                            reference.motors[action]]
                        let body = jointBodies[
                            environment * armActionCount
                                + side * 4 + jointOffset]
                        let axis = body.rotation.act(joint.hingeAxis!)
                        let anchor = body.position
                            + body.rotation.act(joint.rB)
                        jacobian.append(
                            cross(axis, hand.position - anchor))
                    }
                    let deltas = Self.dampedLeastSquaresJointDelta(
                        jacobian: jacobian, taskDelta: taskDelta,
                        damping: 0.03)
                    let normalizedCorrections = (0..<4).map { jointOffset in
                        let action = firstJoint + jointOffset
                        return simd_clamp(
                            deltas[jointOffset]
                                / HumanoidWalkEnv.actionScales[action],
                            -feedback.maximumActionCorrection,
                            feedback.maximumActionCorrection)
                    }
                    diagnostics.append(GraspFeedbackArmDiagnostic(
                        taskDelta: taskDelta,
                        unclampedTaskDeltaMagnitude:
                            unclampedTaskMagnitude,
                        jointDeltaRadians: deltas,
                        normalizedActionCorrection: normalizedCorrections))
                }
                return diagnostics
            }
        }

        func graspFeedbackArmCorrections(
            task: HumanoidBoxCarryTask,
            feedback: ObjectRelativeGraspFeedback
        ) -> [[Float]] {
            graspFeedbackArmDiagnostics(
                task: task, feedback: feedback
            ).map { environment in
                environment.flatMap { $0.normalizedActionCorrection }
            }
        }

        func applyTrajectory(
            _ parameters: [[Float]], step: Int, denominator: Int,
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            forwardOnlyBaseCommand: Bool = false,
            holonomicBaseCommand: Bool = false,
            locomotionCheckpointDirectory: String? = nil,
            locomotionCommandSpeed: Float? = nil,
            applyLegAndTorsoTrajectory: Bool = true,
            trajectoryStart: StructuredTrajectoryBoundary? = nil,
            graspFeedback: ObjectRelativeGraspFeedback? = nil
        ) throws -> RLActionBatch {
            var actions = try policyActions(task: task, observation: observation)
            let graspFeedbackCorrections = graspFeedback.map {
                graspFeedbackArmCorrections(task: task, feedback: $0)
            }
            let baseLegActions = applyLegAndTorsoTrajectory
                    && configuration.legBlendKnotCount > 0
                ? try baseLegPolicyActions(
                    task: task, observation: observation,
                    forwardOnlyCommand: forwardOnlyBaseCommand,
                    holonomicCommand: holonomicBaseCommand,
                    locomotionCheckpointDirectory:
                        locomotionCheckpointDirectory,
                    locomotionCommandSpeed: locomotionCommandSpeed) : nil
            let progress = Float(step + 1) / Float(denominator)
            for environment in parameters.indices {
                var delta = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(parameters[environment].prefix(armParameterCount)),
                    knotCount: configuration.trajectoryKnotCount,
                    progress: progress)
                if let trajectoryStart {
                    for action in delta.indices {
                        delta[action] = continuedTrajectoryValue(
                            zeroStartedValue: delta[action],
                            initialValue: trajectoryStart.armDelta[action],
                            progress: progress,
                            knotCount: configuration.trajectoryKnotCount)
                    }
                }
                let base = environment * task.spec.action.elementCount
                if let baseLegActions {
                    var blend = legBlendFraction(
                        parameters[environment], progress: progress,
                        armParameterCount: armParameterCount,
                        knotCount: configuration.legBlendKnotCount)
                    if let trajectoryStart {
                        blend = continuedTrajectoryValue(
                            zeroStartedValue: blend,
                            initialValue: trajectoryStart.legBlend,
                            progress: progress,
                            knotCount: configuration.legBlendKnotCount)
                    }
                    blend = simd_clamp(blend, 0, 1)
                    for action in 0..<10 {
                        let index = base + action
                        actions.values[index] = (1 - blend)
                            * actions.values[index]
                            + blend * baseLegActions.values[index]
                    }
                }
                if applyLegAndTorsoTrajectory
                    && configuration.legResidualKnotCount > 0 {
                    for action in 0..<10 {
                        var residual = legResidualAction(
                            parameters[environment],
                            action: action, progress: progress,
                            armParameterCount: armParameterCount,
                            blendKnotCount:
                                configuration.legBlendKnotCount,
                            residualKnotCount:
                                configuration.legResidualKnotCount,
                            maximumAction: configuration
                                .maximumLegResidualAction)
                        if let trajectoryStart {
                            residual = continuedTrajectoryValue(
                                zeroStartedValue: residual,
                                initialValue:
                                    trajectoryStart.legResidual[action],
                                progress: progress,
                                knotCount:
                                    configuration.legResidualKnotCount)
                        }
                        actions.values[base + action] = simd_clamp(
                            actions.values[base + action]
                                + residual,
                            -0.999, 0.999)
                    }
                }
                if applyLegAndTorsoTrajectory
                    && configuration.torsoResidualKnotCount > 0 {
                    var residual = torsoResidualAction(
                        parameters[environment], progress: progress,
                        armParameterCount: armParameterCount,
                        blendKnotCount:
                            configuration.legBlendKnotCount,
                        legResidualKnotCount:
                            configuration.legResidualKnotCount,
                        torsoResidualKnotCount:
                            configuration.torsoResidualKnotCount,
                        maximumAction: configuration
                            .maximumTorsoResidualAction)
                    if let trajectoryStart {
                        residual = continuedTrajectoryValue(
                            zeroStartedValue: residual,
                            initialValue: trajectoryStart.torsoResidual,
                            progress: progress,
                            knotCount:
                                configuration.torsoResidualKnotCount)
                    }
                    actions.values[base + 10] = simd_clamp(
                        actions.values[base + 10]
                            + residual,
                        -0.999, 0.999)
                }
                for arm in 0..<armActionCount {
                    let index = base + firstArmAction + arm
                    actions.values[index] = simd_clamp(
                        actions.values[index] + delta[arm], -0.999, 0.999)
                }
                if configuration.armAsymmetryKnotCount > 0 {
                    for arm in 0..<4 {
                        var correction = armAsymmetryAction(
                            parameters[environment], action: arm,
                            progress: progress,
                            armParameterCount: armParameterCount,
                            blendKnotCount:
                                configuration.legBlendKnotCount,
                            legResidualKnotCount:
                                configuration.legResidualKnotCount,
                            torsoResidualKnotCount:
                                configuration.torsoResidualKnotCount,
                            asymmetryKnotCount:
                                configuration.armAsymmetryKnotCount,
                            maximumAction:
                                configuration.maximumArmAsymmetryAction)
                        if let trajectoryStart {
                            correction = continuedTrajectoryValue(
                                zeroStartedValue: correction,
                                initialValue:
                                    trajectoryStart.armAsymmetry[arm],
                                progress: progress,
                                knotCount:
                                    configuration.armAsymmetryKnotCount)
                        }
                        actions.values[base + firstArmAction + arm] =
                            simd_clamp(actions.values[
                                base + firstArmAction + arm] + correction,
                                -0.999, 0.999)
                        actions.values[base + firstArmAction + 4 + arm] =
                            simd_clamp(actions.values[
                                base + firstArmAction + 4 + arm] - correction,
                                -0.999, 0.999)
                    }
                }
                if let graspFeedback, let graspFeedbackCorrections {
                    for arm in 0..<armActionCount {
                        let index = base + firstArmAction + arm
                        actions.values[index] = simd_clamp(
                            actions.values[index]
                                + graspFeedback.blend
                                    * graspFeedbackCorrections[
                                        environment][arm],
                            -0.999, 0.999)
                    }
                }
            }
            return actions
        }

        func states(_ task: HumanoidBoxCarryTask) -> [State] {
            zip(task.environment.states(),
                task.environment.manipulationStates()).map {
                    State(humanoid: $0.0, manipulation: $0.1)
                }
        }

        func flags(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            result: RLStepBatch
        ) -> [Flags] {
            let humanoids = task.environment.states()
            let manipulation = task.environment.manipulationStates()
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            let loadBearing = result.metrics[
                "state/friction_load_bearing_grasp"]!
            let frictionSupport = result.metrics[
                "state/grasp_friction_support_fraction"]!
            let graspQuality = result.metrics["state/grasp_quality"]!
            let pedestal = result.metrics["state/box_pedestal_contact"]!
            let destination = result.metrics[
                "state/box_destination_contact"]!
            let ground = result.metrics["state/box_ground_contact"]!
            return (0..<task.spec.numEnvironments).map { environment in
                let rootUp = humanoids[environment].root.rotation
                    .act(F3(0, 0, 1)).z
                let boxUp = manipulation[environment].object.rotation
                    .act(F3(0, 0, 1)).z
                return Flags(
                    bilateral: left[environment] > 0.5
                        && right[environment] > 0.5,
                    loadBearingGrasp: loadBearing[environment] > 0.5,
                    graspFrictionSupportFraction:
                        frictionSupport[environment],
                    graspQuality: graspQuality[environment],
                    unsupported: isUnsupported(
                        pedestalContact: pedestal[environment],
                        destinationContact: destination[environment],
                        groundContact: ground[environment]),
                    physicallyLifted: observation.policy[
                        environment
                            * HumanoidBoxCarryTask.observationDimension + 89] > 0.5,
                    robotUpright: rootUp > 0.75,
                    boxUpright: boxUp > 0.75,
                    failed: result.terminated[environment]
                        || result.truncated[environment])
            }
        }

        struct PortableSource {
            var state: HumanoidBoxCarryTask.PortableSpeculationState
            var observation: RLObservationBatch
            var result: RLStepBatch
            var sourceCount: Int
        }

        var sourceCache = [Int: CachedSource]()
        var portableSource: PortableSource?
        var sourceAppliedActions: [[Float]]?

        func replicatedObservation(
            _ source: RLObservationBatch, sourceEnvironment: Int,
            sourceCount: Int, spec: RLTaskSpec
        ) -> RLObservationBatch {
            precondition((0..<sourceCount).contains(sourceEnvironment))
            var replicated = RLObservationBatch(spec: spec)

            func copyRows(
                _ values: ContiguousArray<Float>,
                into destination: inout ContiguousArray<Float>
            ) {
                guard !values.isEmpty else { return }
                precondition(values.count.isMultiple(of: sourceCount))
                let width = values.count / sourceCount
                let start = sourceEnvironment * width
                let row = values[start..<(start + width)]
                destination.removeAll(keepingCapacity: true)
                destination.reserveCapacity(spec.numEnvironments * width)
                for _ in 0..<spec.numEnvironments {
                    destination.append(contentsOf: row)
                }
            }

            copyRows(source.policy, into: &replicated.policy)
            copyRows(source.privileged, into: &replicated.privileged)
            return replicated
        }

        func replicatedResult(
            _ source: RLStepBatch, sourceEnvironment: Int,
            sourceCount: Int, spec: RLTaskSpec
        ) -> RLStepBatch {
            precondition((0..<sourceCount).contains(sourceEnvironment))
            var replicated = RLStepBatch(spec: spec)
            replicated.observations = replicatedObservation(
                source.observations, sourceEnvironment: sourceEnvironment,
                sourceCount: sourceCount, spec: spec)
            replicated.rewards = ContiguousArray(
                repeating: source.rewards[sourceEnvironment],
                count: spec.numEnvironments)
            replicated.terminated = ContiguousArray(
                repeating: source.terminated[sourceEnvironment],
                count: spec.numEnvironments)
            replicated.truncated = ContiguousArray(
                repeating: source.truncated[sourceEnvironment],
                count: spec.numEnvironments)
            replicated.successes = ContiguousArray(
                repeating: source.successes[sourceEnvironment],
                count: spec.numEnvironments)
            replicated.imitationMilestones = ContiguousArray(
                repeating: source.imitationMilestones[sourceEnvironment],
                count: spec.numEnvironments)
            replicated.hasFinalObservation = ContiguousArray(
                repeating: source.hasFinalObservation[sourceEnvironment],
                count: spec.numEnvironments)
            if !source.finalObservations.isEmpty {
                let width = source.finalObservations.count / sourceCount
                let start = sourceEnvironment * width
                let row = source.finalObservations[start..<(start + width)]
                replicated.finalObservations.removeAll(keepingCapacity: true)
                replicated.finalObservations.reserveCapacity(
                    spec.numEnvironments * width)
                for _ in 0..<spec.numEnvironments {
                    replicated.finalObservations.append(contentsOf: row)
                }
            }
            replicated.metrics = source.metrics.mapValues { values in
                ContiguousArray(
                    repeating: values[sourceEnvironment],
                    count: spec.numEnvironments)
            }
            return replicated
        }

        func canonicalizeSpeculationRows(
            runtime: inout WarmTask, count: Int
        ) {
            guard count > 1 else { return }
            runtime.task.canonicalizeSpeculationReplicas(
                observation: &runtime.observation,
                result: &runtime.result)
        }

        func prepareSource(count: Int) throws -> WarmTask {
            if var cached = sourceCache[count] {
                cached.runtime.task.restoreSpeculationSnapshot(
                    cached.snapshot)
                cached.runtime.simulatedPreparationSteps = 0
                return cached.runtime
            }
            if let portableSource,
               count != portableSource.sourceCount {
                var runtime = try makeWarmTask(count: count)
                runtime.task.restorePortableSpeculationState(
                    portableSource.state)
                runtime.observation = replicatedObservation(
                    portableSource.observation, sourceEnvironment: 0,
                    sourceCount: portableSource.sourceCount,
                    spec: runtime.task.spec)
                runtime.result = replicatedResult(
                    portableSource.result, sourceEnvironment: 0,
                    sourceCount: portableSource.sourceCount,
                    spec: runtime.task.spec)
                let restoredFlags = flags(
                    task: runtime.task, observation: runtime.observation,
                    result: runtime.result)
                let successes = restoredFlags.filter {
                    $0.bilateral && $0.unsupported && $0.physicallyLifted
                        && $0.robotUpright && $0.boxUpright && !$0.failed
                }.count
                runtime.sourceReplaySuccessFraction =
                    Float(successes) / Float(restoredFlags.count)
                guard runtime.sourceReplaySuccessFraction >= 0.8 else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "portable source boundary failed physical validation")
                }
                sourceCache[count] = CachedSource(
                    runtime: runtime,
                    snapshot:
                        runtime.task.captureSpeculationSnapshot())
                return runtime
            }
            var runtime = try makeWarmTask(count: count)
            var executedSourceActions = [[Float]]()
            executedSourceActions.reserveCapacity(sourceControlSteps)
            for (stageIndex, stage) in sourceStages.enumerated() {
                let stageTrajectoryStart: StructuredTrajectoryBoundary?
                if stage.continueFromPreviousTrajectoryTerminal == true {
                    guard let boundary = sourceStages[..<stageIndex]
                            .reversed()
                            .compactMap({
                                terminalBoundary(for: $0)
                            }).first else {
                        throw RLEnvironmentError.invalidConfiguration(
                            "continued source stage has no preceding trajectory")
                    }
                    stageTrajectoryStart = boundary
                } else {
                    stageTrajectoryStart = nil
                }
                if stage.canonicalizeReplicasBeforeExecution == true {
                    canonicalizeSpeculationRows(
                        runtime: &runtime, count: count)
                }
                let requiresCertification =
                    stage.minimumCarryDistanceMeters != nil
                    || stage.minimumDestinationProgressMeters != nil
                    || stage.minimumRootDestinationProgressMeters != nil
                    || stage.minimumTouchdowns != nil
                    || stage.minimumAlternatingSteps != nil
                    || stage.minimumSwingFootLiftMeters != nil
                    || stage.minimumFootAirTimeSeconds != nil
                    || stage.minimumFootUnloadingFraction != nil
                    || stage.minimumTerminalFootUnloadingFraction != nil
                    || stage.minimumClearanceMeters != nil
                    || stage.maximumPathDownwardBoxVelocityMPS != nil
                    || stage.minimumGraspQuality != nil
                    || stage.certificationDwellSteps != nil
                let requiredDwell = stage.certificationDwellSteps ?? 1
                guard requiredDwell > 0,
                      (stage.minimumCarryDistanceMeters.map {
                          $0.isFinite && $0 >= 0
                      } ?? true),
                      (stage.minimumDestinationProgressMeters.map {
                          $0.isFinite && $0 >= 0
                      } ?? true),
                      (stage.minimumRootDestinationProgressMeters.map {
                          $0.isFinite && $0 >= 0
                      } ?? true),
                      (stage.minimumTouchdowns.map { $0 >= 0 } ?? true),
                      (stage.minimumAlternatingSteps.map { $0 >= 0 }
                        ?? true),
                      (stage.minimumSwingFootLiftMeters.map {
                          $0.isFinite && $0 >= 0 && $0 <= 0.25
                      } ?? true),
                      (stage.minimumFootAirTimeSeconds.map {
                          $0.isFinite && $0 >= 0 && $0 <= 1
                      } ?? true),
                      (stage.minimumFootUnloadingFraction.map {
                          $0.isFinite && (0...1).contains($0)
                      } ?? true),
                      (stage.minimumTerminalFootUnloadingFraction.map {
                          $0.isFinite && (0...1).contains($0)
                      } ?? true),
                      (stage.minimumClearanceMeters.map {
                          $0.isFinite && $0 >= 0.01
                      } ?? true),
                      (stage.maximumPathDownwardBoxVelocityMPS.map {
                          $0.isFinite && $0 >= 0
                      } ?? true),
                      (stage.minimumGraspQuality.map {
                          $0.isFinite && (0...1).contains($0)
                      } ?? true) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "invalid certified source physical-flow stage")
                }
                var certificationDwells = [Int](
                    repeating: 0, count: count)
                let stageInitialPlacementDistances = Array(
                    runtime.result.metrics["state/placement_distance_m"]!)
                let stageInitialRootDestinationDistances = Array(
                    runtime.result.metrics[
                        "state/root_destination_distance_m"]!)
                let stageInitialAlternatingSteps = Array(
                    runtime.result.metrics[
                        "state/loaded_alternating_steps"]!)
                let stageInitialTouchdowns = Array(
                    runtime.result.metrics["state/loaded_touchdowns"]!)
                var stageMaximumFootUnloadingFractions = Array(
                    runtime.result.metrics[
                        "state/foot_unloading_fraction"]!)
                var stageMaximumLoadedFootAirTimes = [Float](
                    repeating: 0, count: count)
                var stageMaximumSwingFootLifts = [Float](
                    repeating: 0, count: count)
                var stagePathVelocitySafe = [Bool](
                    repeating: true, count: count)
                let stageGraspFeedback =
                    stage.graspAnchorFeedbackBlend.map {
                        ObjectRelativeGraspFeedback(
                            blend: $0,
                            velocityHorizonSeconds:
                                stage
                                    .graspAnchorFeedbackVelocityHorizonSeconds!,
                            maximumActionCorrection:
                                stage
                                    .graspAnchorFeedbackMaximumActionCorrection!,
                            inwardPreloadMeters:
                                stage
                                    .graspAnchorFeedbackInwardPreloadMeters!,
                            leftAnchorBoxLocal: F3(
                                stage.leftGraspAnchorBoxLocalMeters![0],
                                stage.leftGraspAnchorBoxLocalMeters![1],
                                stage.leftGraspAnchorBoxLocalMeters![2]),
                            rightAnchorBoxLocal: F3(
                                stage.rightGraspAnchorBoxLocalMeters![0],
                                stage.rightGraspAnchorBoxLocalMeters![1],
                                stage.rightGraspAnchorBoxLocalMeters![2]),
                            boxHeightMeters:
                                stage.graspAnchorBoxHeightMeters!)
                    }
                for step in 0..<stage.controlSteps {
                    let actions: RLActionBatch
                    if let committed =
                            stage.appliedNormalizedActions?[step] {
                        var values = ContiguousArray<Float>()
                        values.reserveCapacity(
                            count * runtime.task.spec.action.elementCount)
                        for _ in 0..<count {
                            values.append(contentsOf: committed)
                        }
                        actions = try RLActionBatch(
                            numEnvironments: count,
                            actionDimension:
                                runtime.task.spec.action.elementCount,
                            values: values)
                    } else if stage.policyOnly == true {
                        actions = try policyActions(
                            task: runtime.task,
                            observation: runtime.observation)
                    } else if let sequence = stage.trajectorySequence {
                        actions = try applyTrajectory(
                            [[Float]](repeating: sequence[step],
                                      count: count),
                            step: stage.trajectoryEvaluationStep(at: step),
                            denominator: stage
                                .trajectorySequenceStepDenominator!,
                            task: runtime.task,
                            observation: runtime.observation,
                            forwardOnlyBaseCommand:
                                stage.forwardOnlyBaseCommand == true,
                            holonomicBaseCommand:
                                stage.holonomicBaseCommand == true,
                            locomotionCheckpointDirectory:
                                stage.locomotionCheckpointDirectory,
                            locomotionCommandSpeed:
                                stage.locomotionCommandSpeed,
                            trajectoryStart: stageTrajectoryStart,
                            graspFeedback: stageGraspFeedback)
                    } else {
                        actions = try applyTrajectory(
                            [[Float]](
                                repeating: stage.trajectory, count: count),
                            step: step,
                            denominator: stage.trajectoryDurationSteps
                                ?? stage.controlSteps,
                            task: runtime.task,
                            observation: runtime.observation,
                            forwardOnlyBaseCommand:
                                stage.forwardOnlyBaseCommand == true,
                            holonomicBaseCommand:
                                stage.holonomicBaseCommand == true,
                            locomotionCheckpointDirectory:
                                stage.locomotionCheckpointDirectory,
                            locomotionCommandSpeed:
                                stage.locomotionCommandSpeed,
                            trajectoryStart: stageTrajectoryStart,
                            graspFeedback: stageGraspFeedback)
                    }
                    executedSourceActions.append(Array(
                        actions.values.prefix(
                            runtime.task.spec.action.elementCount)))
                    try runtime.task.step(
                        actions: actions, into: &runtime.result)
                    runtime.observation = runtime.result.observations
                    guard requiresCertification else { continue }
                    let currentStates = states(runtime.task)
                    let currentFlags = flags(
                        task: runtime.task,
                        observation: runtime.observation,
                        result: runtime.result)
                    let carry = runtime.result.metrics[
                        "state/carry_distance_m"]!
                    let clearance = runtime.result.metrics[
                        "state/box_clearance_m"]!
                    let placement = runtime.result.metrics[
                        "state/placement_distance_m"]!
                    let rootDestinationDistance = runtime.result.metrics[
                        "state/root_destination_distance_m"]!
                    let alternatingSteps = runtime.result.metrics[
                        "state/loaded_alternating_steps"]!
                    let touchdowns = runtime.result.metrics[
                        "state/loaded_touchdowns"]!
                    let loadedFootAirTimes = runtime.result.metrics[
                        "state/maximum_loaded_foot_air_time_s"]!
                    let swingFootLifts = runtime.result.metrics[
                        "state/maximum_loaded_swing_clearance_m"]!
                    let footUnloadingFractions = runtime.result.metrics[
                        "state/foot_unloading_fraction"]!
                    for environment in currentFlags.indices {
                        stageMaximumFootUnloadingFractions[environment] = max(
                            stageMaximumFootUnloadingFractions[environment],
                            footUnloadingFractions[environment])
                        stageMaximumLoadedFootAirTimes[environment] = max(
                            stageMaximumLoadedFootAirTimes[environment],
                            loadedFootAirTimes[environment])
                        stageMaximumSwingFootLifts[environment] = max(
                            stageMaximumSwingFootLifts[environment],
                            swingFootLifts[environment])
                        let rootUp = currentStates[environment].humanoid.root
                            .rotation.act(F3(0, 0, 1)).z
                        let boxUp = currentStates[environment].manipulation
                            .object.rotation.act(F3(0, 0, 1)).z
                        let physical =
                            currentFlags[environment].loadBearingGrasp
                            && currentFlags[environment].unsupported
                            && currentFlags[environment].physicallyLifted
                            && rootUp > 0.9 && boxUp > 0.9
                            && !currentFlags[environment].failed
                        let carryPassed = stage.minimumCarryDistanceMeters.map {
                            carry[environment] >= $0
                        } ?? true
                        let destinationProgressPassed = stage
                            .minimumDestinationProgressMeters.map {
                                minimumDestinationProgressPassed(
                                    stageInitialPlacementDistances[environment]
                                        - placement[environment],
                                    minimum: $0)
                            } ?? true
                        let clearancePassed = stage.minimumClearanceMeters.map {
                            clearance[environment] >= $0
                        } ?? true
                        let rootProgressPassed = stage
                            .minimumRootDestinationProgressMeters.map {
                                stageInitialRootDestinationDistances[
                                    environment]
                                    - rootDestinationDistance[environment]
                                    >= $0
                            } ?? true
                        let stepsPassed = stage.minimumAlternatingSteps.map {
                            Int(alternatingSteps[environment]
                                - stageInitialAlternatingSteps[environment])
                                >= $0
                        } ?? true
                        let touchdownsPassed = stage.minimumTouchdowns.map {
                            Int(touchdowns[environment]
                                - stageInitialTouchdowns[environment]) >= $0
                        } ?? true
                        let footAirTimePassed =
                            stage.minimumFootAirTimeSeconds.map {
                                stageMaximumLoadedFootAirTimes[environment]
                                    >= $0
                            } ?? true
                        let swingFootLiftPassed =
                            stage.minimumSwingFootLiftMeters.map {
                                stageMaximumSwingFootLifts[environment]
                                    >= $0
                            } ?? true
                        let footUnloadingPassed =
                            stage.minimumFootUnloadingFraction.map {
                                stageMaximumFootUnloadingFractions[
                                    environment] >= $0
                            } ?? true
                        let terminalFootUnloadingPassed = stage
                            .minimumTerminalFootUnloadingFraction.map {
                                footUnloadingFractions[environment] >= $0
                            } ?? true
                        let pathVelocityPassed = stage
                            .maximumPathDownwardBoxVelocityMPS.map {
                                currentStates[environment].manipulation.object
                                    .linearVelocity.z >= -$0
                            } ?? true
                        stagePathVelocitySafe[environment] =
                            stagePathVelocitySafe[environment]
                                && pathVelocityPassed
                        let graspPassed = stage.minimumGraspQuality.map {
                            currentFlags[environment].graspQuality >= $0
                        } ?? true
                        certificationDwells[environment] = physical
                                && carryPassed && destinationProgressPassed
                                && rootProgressPassed && stepsPassed
                                && touchdownsPassed
                                && swingFootLiftPassed
                                && footAirTimePassed
                                && footUnloadingPassed
                                && terminalFootUnloadingPassed
                                && clearancePassed
                                && stagePathVelocitySafe[environment]
                                && graspPassed
                            ? certificationDwells[environment] + 1 : 0
                    }
                }
                if requiresCertification {
                    let successes = certificationDwells.filter {
                        $0 >= requiredDwell
                    }.count
                    guard Float(successes) / Float(count) >= 0.8 else {
                        let carry = runtime.result.metrics[
                            "state/carry_distance_m"]!
                        let clearance = runtime.result.metrics[
                            "state/box_clearance_m"]!
                        let grasp = runtime.result.metrics[
                            "state/grasp_quality"]!
                        let terminalFootUnloading = runtime.result.metrics[
                            "state/foot_unloading_fraction"]!
                        let maximumDwell = certificationDwells.max() ?? 0
                        throw RLEnvironmentError.invalidConfiguration(
                            "certified source flow stage \(stageIndex + 1)/\(sourceStages.count) "
                                + "failed its endpoint requirements "
                                + "(successes \(successes)/\(count), "
                                + "maximum dwell \(maximumDwell)/\(requiredDwell), "
                                + "terminal carry \(carry.min() ?? .nan)...\(carry.max() ?? .nan) m, "
                                + "clearance \(clearance.min() ?? .nan)...\(clearance.max() ?? .nan) m, "
                                + "grasp \(grasp.min() ?? .nan)...\(grasp.max() ?? .nan), "
                                + "maximum foot air "
                                + "\(stageMaximumLoadedFootAirTimes.max() ?? .nan) s, "
                                + "maximum swing lift "
                                + "\(stageMaximumSwingFootLifts.max() ?? .nan) m, "
                                + "maximum unloading "
                                + "\(stageMaximumFootUnloadingFractions.max() ?? .nan), "
                                + "terminal unloading "
                                + "\(terminalFootUnloading.min() ?? .nan)..."
                                + "\(terminalFootUnloading.max() ?? .nan))")
                    }
                }
            }
            canonicalizeSpeculationRows(runtime: &runtime, count: count)
            if sourceAppliedActions == nil, !executedSourceActions.isEmpty {
                sourceAppliedActions = executedSourceActions
            }
            if portableSource == nil, !sourceStages.isEmpty {
                portableSource = PortableSource(
                    state: runtime.task.capturePortableSpeculationState(),
                    observation: runtime.observation,
                    result: runtime.result,
                    sourceCount: count)
            }
            if sourceStages.isEmpty {
                runtime.simulatedPreparationSteps = runtime.warmupSteps
                sourceCache[count] = CachedSource(
                    runtime: runtime,
                    snapshot: runtime.task.captureSpeculationSnapshot())
                return runtime
            }
            let sourceFlags = flags(
                task: runtime.task, observation: runtime.observation,
                result: runtime.result)
            let successes = sourceFlags.filter {
                $0.bilateral && $0.unsupported && $0.physicallyLifted
                    && $0.robotUpright && $0.boxUpright && !$0.failed
            }.count
            runtime.sourceReplaySuccessFraction = Float(successes)
                / Float(sourceFlags.count)
            guard runtime.sourceReplaySuccessFraction >= 0.8 else {
                throw RLEnvironmentError.invalidConfiguration(
                    String(format:
                        "source physical flow replay passed only %.1f%% of replicas",
                        100 * runtime.sourceReplaySuccessFraction))
            }
            runtime.simulatedPreparationSteps = runtime.warmupSteps
                + sourceControlSteps
            sourceCache[count] = CachedSource(
                runtime: runtime,
                snapshot: runtime.task.captureSpeculationSnapshot())
            return runtime
        }

        let targetSelectionClearance = configuration
            .targetDiscoveryObjectiveClearanceMeters
            ?? max(0.04, configuration.minimumTargetClearanceMeters)
        let targetDiscoveryClearance = configuration
            .targetDiscoveryObjectiveClearanceMeters
            ?? max(0.025, configuration.minimumTargetClearanceMeters)
        let targetSelectionDestinationProgress = configuration
            .targetDiscoveryObjectiveDestinationProgressMeters
            ?? configuration.minimumTargetDestinationProgressMeters
        let targetSelectionRootDestinationProgress = configuration
            .targetDiscoveryObjectiveRootDestinationProgressMeters
            ?? configuration.minimumTargetRootDestinationProgressMeters
        let targetDiscoverySwingFootLift = configuration
            .targetDiscoveryObjectiveSwingFootLiftMeters ?? 0
        let targetDiscoveryFootAirTime = configuration
            .targetDiscoveryObjectiveFootAirTimeSeconds
            ?? configuration.minimumTargetFootAirTimeSeconds
        let targetDiscoveryFootUnloading = configuration
            .targetDiscoveryObjectiveFootUnloadingFraction
            ?? configuration.minimumTargetFootUnloadingFraction
        let targetDiscoveryGraspQuality = configuration
            .targetDiscoveryObjectiveGraspQuality
            ?? configuration.minimumTargetGraspQuality

        func generateTarget(using trajectory: [Float]) throws -> Target {
            let count = min(32, configuration.populationSize)
            var runtime = try prepareSource(count: count)
            let graspFeedback = configuration.graspAnchorFeedbackBlend > 0
                ? captureGraspFeedback(
                    task: runtime.task,
                    blend: configuration.graspAnchorFeedbackBlend)
                : nil
            let initialPlacementDistances = Array(
                runtime.result.metrics["state/placement_distance_m"]!)
            let initialRootDestinationDistances = Array(
                runtime.result.metrics[
                    "state/root_destination_distance_m"]!)
            let initialAlternatingSteps = Array(
                runtime.result.metrics[
                    "state/loaded_alternating_steps"]!)
            let initialTouchdowns = Array(
                runtime.result.metrics["state/loaded_touchdowns"]!)
            var best: Target?
            var maximumClearance: Float = -.infinity
            var maximumCarryDistance: Float = 0
            var maximumStableCarryDistance: Float = 0
            var maximumDestinationProgress: Float = -.infinity
            var maximumRootDestinationProgress: Float = -.infinity
            var maximumLoadedAlternatingSteps = 0
            var maximumLoadedTouchdowns = 0
            var maximumSwingFootLift: Float = 0
            var maximumLoadedFootAirTime: Float = 0
            var maximumFootUnloadingFraction: Float = 0
            var stableCarryPath = true
            var feasibilityDwell = 0
            var maximumFeasibilityDwell = 0
            var predicateDwells = [Int](repeating: 0, count: 9)
            var maximumPredicateDwells = [Int](repeating: 0, count: 9)
            var maximumGraspQuality: Float = -.infinity
            var firstStablePathViolationStep: Int?
            let repeated = [[Float]](
                repeating: trajectory, count: count)
            for step in 0..<targetExecutionSteps {
                let actions = try applyTrajectory(
                    repeated, step: step,
                    denominator: configuration.targetGenerationSteps,
                    task: runtime.task, observation: runtime.observation,
                    trajectoryStart: currentTrajectoryStart,
                    graspFeedback: graspFeedback)
                try runtime.task.step(actions: actions, into: &runtime.result)
                runtime.observation = runtime.result.observations
                let allStates = states(runtime.task)
                let allFlags = flags(
                    task: runtime.task, observation: runtime.observation,
                    result: runtime.result)
                maximumGraspQuality = max(
                    maximumGraspQuality, allFlags[0].graspQuality)
                let clearances = runtime.result.metrics[
                    "state/box_clearance_m"]!
                let carryDistances = runtime.result.metrics[
                    "state/carry_distance_m"]!
                let placementDistances = runtime.result.metrics[
                    "state/placement_distance_m"]!
                let rootDestinationDistances = runtime.result.metrics[
                    "state/root_destination_distance_m"]!
                let alternatingSteps = runtime.result.metrics[
                    "state/loaded_alternating_steps"]!
                let touchdowns = runtime.result.metrics[
                    "state/loaded_touchdowns"]!
                let loadedFootAirTimes = runtime.result.metrics[
                    "state/maximum_loaded_foot_air_time_s"]!
                let footUnloadingFractions = runtime.result.metrics[
                    "state/foot_unloading_fraction"]!
                let destinationProgress = initialPlacementDistances[0]
                    - placementDistances[0]
                let rootDestinationProgress =
                    initialRootDestinationDistances[0]
                        - rootDestinationDistances[0]
                let loadedAlternatingSteps = max(
                    0, Int(alternatingSteps[0]
                        - initialAlternatingSteps[0]))
                let loadedTouchdowns = max(
                    0, Int(touchdowns[0] - initialTouchdowns[0]))
                let swingFootLift = runtime.result.metrics[
                    "state/maximum_loaded_swing_clearance_m"]![0]
                maximumClearance = max(maximumClearance, clearances[0])
                maximumCarryDistance = max(
                    maximumCarryDistance, carryDistances[0])
                maximumDestinationProgress = max(
                    maximumDestinationProgress, destinationProgress)
                maximumRootDestinationProgress = max(
                    maximumRootDestinationProgress,
                    rootDestinationProgress)
                maximumLoadedAlternatingSteps = max(
                    maximumLoadedAlternatingSteps,
                    loadedAlternatingSteps)
                maximumLoadedTouchdowns = max(
                    maximumLoadedTouchdowns, loadedTouchdowns)
                maximumSwingFootLift = max(
                    maximumSwingFootLift, swingFootLift)
                maximumLoadedFootAirTime = max(
                    maximumLoadedFootAirTime, loadedFootAirTimes[0])
                maximumFootUnloadingFraction = max(
                    maximumFootUnloadingFraction,
                    footUnloadingFractions[0])
                let boxUp = allStates[0].manipulation.object.rotation
                    .act(F3(0, 0, 1)).z
                let robotUp = allStates[0].humanoid.root.rotation
                    .act(F3(0, 0, 1)).z
                let stableCarryManifold = allFlags[0].loadBearingGrasp
                    && allFlags[0].unsupported
                    && allFlags[0].physicallyLifted
                    // The bridge target should be visually and mechanically
                    // useful, not merely below the looser fall guard used for
                    // candidate rejection.
                    && robotUp > 0.9
                    && boxUp > 0.9
                    && !allFlags[0].failed
                    && clearances[0] >= 0.01
                    && pathDownwardVelocityFeasible(
                        boxVerticalVelocity: allStates[0].manipulation.object
                            .linearVelocity.z,
                        maximumDownwardVelocity: configuration
                            .maximumTargetPathDownwardBoxVelocityMPS)
                let predicatePasses = [
                    allFlags[0].bilateral,
                    allFlags[0].unsupported,
                    allFlags[0].physicallyLifted,
                    robotUp > 0.9 && boxUp > 0.9
                        && !allFlags[0].failed,
                    carryDistances[0] >= configuration
                        .minimumTargetCarryDistanceMeters,
                    minimumDestinationProgressPassed(
                        destinationProgress,
                        minimum: configuration
                            .minimumTargetDestinationProgressMeters),
                    clearances[0] >= configuration
                        .minimumTargetClearanceMeters,
                    allFlags[0].graspQuality
                        >= configuration.minimumTargetGraspQuality,
                    allFlags[0].loadBearingGrasp,
                ]
                for index in predicatePasses.indices {
                    predicateDwells[index] = predicatePasses[index]
                        ? predicateDwells[index] + 1 : 0
                    maximumPredicateDwells[index] = max(
                        maximumPredicateDwells[index],
                        predicateDwells[index])
                }
                if !stableCarryManifold,
                   firstStablePathViolationStep == nil {
                    firstStablePathViolationStep = step + 1
                }
                stableCarryPath = stableCarryPath && stableCarryManifold
                if stableCarryManifold {
                    maximumStableCarryDistance = max(
                        maximumStableCarryDistance, carryDistances[0])
                }
                let instantFeasible = stableCarryManifold
                    && (!configuration.requireStableCarryPath
                        || stableCarryPath)
                    && carryDistances[0]
                        >= configuration.minimumTargetCarryDistanceMeters
                    && minimumDestinationProgressPassed(
                        destinationProgress,
                        minimum: configuration
                            .minimumTargetDestinationProgressMeters)
                    && rootDestinationProgress >= configuration
                        .minimumTargetRootDestinationProgressMeters
                    && loadedTouchdowns
                        >= configuration.minimumTargetTouchdowns
                    && loadedAlternatingSteps >= configuration
                        .minimumTargetAlternatingSteps
                    && maximumSwingFootLift >= configuration
                        .minimumTargetSwingFootLiftMeters
                    && maximumLoadedFootAirTime >= configuration
                        .minimumTargetFootAirTimeSeconds
                    && maximumFootUnloadingFraction >= configuration
                        .minimumTargetFootUnloadingFraction
                    && (configuration
                        .minimumTargetTerminalFootUnloadingFraction.map {
                            footUnloadingFractions[0] >= $0
                        } ?? true)
                    && clearances[0]
                        >= configuration.minimumTargetClearanceMeters
                    && allFlags[0].graspQuality
                        >= configuration.minimumTargetGraspQuality
                feasibilityDwell = instantFeasible
                    ? feasibilityDwell + 1 : 0
                maximumFeasibilityDwell = max(
                    maximumFeasibilityDwell, feasibilityDwell)
                let feasible = instantFeasible
                    && feasibilityDwell
                        >= configuration.targetFeasibilityDwellSteps
                    && (configuration.targetSelectionStep.map {
                        $0 == step + 1
                    } ?? true)
                guard feasible else { continue }
                let object = allStates[0].manipulation.object
                let root = allStates[0].humanoid.root
                let selection = PhysicalFlowBalancedObjective.evaluate(
                    normalizedErrors: [
                        max(targetSelectionClearance - clearances[0], 0)
                            / max(targetSelectionClearance, 0.02),
                        length(object.linearVelocity) / 0.15,
                        length(object.angularVelocity) / 0.50,
                        length(root.linearVelocity) / 0.30,
                        length(root.angularVelocity) / 0.60,
                        max(0.9 - boxUp, 0) / 0.15,
                        max(0.9 - robotUp, 0) / 0.15,
                        max(configuration.minimumTargetCarryDistanceMeters
                            - carryDistances[0], 0)
                            / max(configuration.minimumTargetCarryDistanceMeters,
                                  0.05),
                        targetSelectionDestinationProgress > 0
                            ? max(targetSelectionDestinationProgress
                                - destinationProgress, 0)
                                / max(targetSelectionDestinationProgress, 0.05)
                            : 0,
                        targetSelectionRootDestinationProgress > 0
                            ? max(targetSelectionRootDestinationProgress
                                - rootDestinationProgress, 0)
                                / max(
                                    targetSelectionRootDestinationProgress,
                                    0.05)
                            : 0,
                        targetDiscoverySwingFootLift > 0
                            ? max(targetDiscoverySwingFootLift
                                - maximumSwingFootLift, 0)
                                / 0.01
                            : 0,
                        targetDiscoveryFootAirTime > 0
                            ? max(targetDiscoveryFootAirTime
                                - maximumLoadedFootAirTime, 0)
                                / max(targetDiscoveryFootAirTime, 0.02)
                            : 0,
                        targetDiscoveryFootUnloading > 0
                            ? max(targetDiscoveryFootUnloading
                                - maximumFootUnloadingFraction, 0)
                                / 0.10
                            : 0,
                        max(targetDiscoveryGraspQuality
                            - allFlags[0].graspQuality, 0)
                            / max(targetDiscoveryGraspQuality, 0.25),
                    ])
                if best == nil || selection.bottleneckLoss
                    < targetSelectionLoss(best!) {
                    best = Target(
                        state: allStates[0], flags: allFlags[0],
                        step: step + 1, clearance: clearances[0],
                        boxUpright: boxUp, robotUpright: robotUp,
                        carryDistance: carryDistances[0],
                        placementDistance: placementDistances[0],
                        destinationProgress: destinationProgress,
                        rootDestinationProgress:
                            rootDestinationProgress,
                        loadedTouchdowns:
                            loadedTouchdowns,
                        loadedAlternatingSteps:
                            loadedAlternatingSteps,
                        maximumSwingFootLift:
                            maximumSwingFootLift,
                        maximumLoadedFootAirTime:
                            maximumLoadedFootAirTime,
                        maximumFootUnloadingFraction:
                            maximumFootUnloadingFraction,
                        terminalFootUnloadingFraction:
                            footUnloadingFractions[0],
                        replicas: allStates, replicaFlags: allFlags,
                        replicaCarryDistances: Array(carryDistances),
                        replicaClearances: Array(clearances),
                        replicaGraspQualities:
                            allFlags.map(\.graspQuality),
                        replicaFootUnloadingFractions:
                            Array(footUnloadingFractions),
                        warmupSteps: runtime.warmupSteps,
                        simulatedPreparationSteps:
                            runtime.simulatedPreparationSteps,
                        sourceReplaySuccessFraction:
                            runtime.sourceReplaySuccessFraction,
                        stableCarryPath: stableCarryPath,
                        graspFeedback: graspFeedback)
                }
            }
            guard let best else {
                throw HumanoidBoxPhysicalFlowTargetFailure(
                    experiment: "humanoid-box-target-discovery-failure-v0",
                    seed: configuration.seed,
                    optimizerSeed: configuration.optimizerSeed,
                    targetGeneratingTrajectory: trajectory,
                    targetGeneratingTrajectorySequence: nil,
                    targetGeneratingTrajectorySequencePhaseSteps: nil,
                    derivedStageContinuesFromSourceTerminal:
                        configuration.continueTrajectoryFromSourceTerminal
                            ? true : nil,
                    targetDiscoveryTiedArmKnots:
                        configuration.targetDiscoveryTiedArmKnots
                            ? true : nil,
                    targetDiscoveryBlockCoordinateSearch:
                        configuration.targetDiscoveryBlockCoordinateSearch
                            ? true : nil,
                    targetDiscoverySeededFromSourceTerminalUpperBody:
                        seedsFromSourceTerminalTrajectory ? true : nil,
                    targetDiscoveryHeldSourceTerminalUpperBody:
                        holdsSourceTerminalUpperBody ? true : nil,
                    targetDiscoveryPreservedProvidedUpperBodySeed:
                        preservesProvidedUpperBodySeed ? true : nil,
                    recedingLocomotionCheckpointDirectory: nil,
                    recedingLocomotionCommandSpeed: nil,
                    targetGenerationSteps:
                        configuration.targetGenerationSteps,
                    targetExecutionSteps: targetExecutionSteps,
                    sourceStages: sourceStages,
                    sourceWarmupAppliedActions:
                        sourceWarmupAppliedActions,
                    sourceAppliedActions: sourceAppliedActions,
                    legBlendKnotCount: configuration.legBlendKnotCount,
                    legResidualKnotCount:
                        configuration.legResidualKnotCount,
                    maximumLegResidualAction:
                        configuration.maximumLegResidualAction,
                    torsoResidualKnotCount:
                        configuration.torsoResidualKnotCount,
                    maximumTorsoResidualAction:
                        configuration.maximumTorsoResidualAction,
                    armAsymmetryKnotCount:
                        configuration.armAsymmetryKnotCount,
                    maximumArmAsymmetryAction:
                        configuration.maximumArmAsymmetryAction,
                    graspAnchorFeedbackBlend:
                        graspFeedback?.blend,
                    graspAnchorFeedbackVelocityHorizonSeconds:
                        graspFeedback?.velocityHorizonSeconds,
                    graspAnchorFeedbackMaximumActionCorrection:
                        graspFeedback?.maximumActionCorrection,
                    graspAnchorFeedbackInwardPreloadMeters:
                        graspFeedback?.inwardPreloadMeters,
                    leftGraspAnchorBoxLocalMeters:
                        graspFeedback.map {
                            [$0.leftAnchorBoxLocal.x,
                             $0.leftAnchorBoxLocal.y,
                             $0.leftAnchorBoxLocal.z]
                        },
                    rightGraspAnchorBoxLocalMeters:
                        graspFeedback.map {
                            [$0.rightAnchorBoxLocal.x,
                             $0.rightAnchorBoxLocal.y,
                             $0.rightAnchorBoxLocal.z]
                        },
                    graspAnchorBoxHeightMeters:
                        graspFeedback?.boxHeightMeters,
                    maximumClearanceMeters: maximumClearance,
                    maximumCarryDistanceMeters: maximumCarryDistance,
                    maximumStableCarryDistanceMeters:
                        maximumStableCarryDistance,
                    maximumDestinationProgressMeters:
                        maximumDestinationProgress,
                    maximumRootDestinationProgressMeters:
                        maximumRootDestinationProgress,
                    maximumLoadedTouchdowns:
                        maximumLoadedTouchdowns,
                    maximumLoadedAlternatingSteps:
                        maximumLoadedAlternatingSteps,
                    maximumSwingFootLiftMeters:
                        maximumSwingFootLift,
                    maximumLoadedFootAirTimeSeconds:
                        maximumLoadedFootAirTime,
                    maximumFootUnloadingFraction:
                        maximumFootUnloadingFraction,
                    maximumFeasibilityDwellSteps:
                        maximumFeasibilityDwell,
                    maximumBilateralDwellSteps:
                        maximumPredicateDwells[0],
                    maximumLoadBearingGraspDwellSteps:
                        maximumPredicateDwells[8],
                    maximumUnsupportedDwellSteps:
                        maximumPredicateDwells[1],
                    maximumPhysicallyLiftedDwellSteps:
                        maximumPredicateDwells[2],
                    maximumUprightDwellSteps:
                        maximumPredicateDwells[3],
                    maximumCarryThresholdDwellSteps:
                        maximumPredicateDwells[4],
                    maximumClearanceThresholdDwellSteps:
                        maximumPredicateDwells[6],
                    maximumGraspQualityThresholdDwellSteps:
                        maximumPredicateDwells[7],
                    maximumGraspQuality: maximumGraspQuality,
                    firstStablePathViolationStep:
                        firstStablePathViolationStep,
                    finalCarryDistanceMeters: nil,
                    finalClearanceMeters: nil,
                    physicalBalanceGatePassed: false,
                    requiredCarryDistanceMeters:
                        configuration.minimumTargetCarryDistanceMeters,
                    requiredDestinationProgressMeters: configuration
                        .minimumTargetDestinationProgressMeters,
                    requiredRootDestinationProgressMeters: configuration
                        .minimumTargetRootDestinationProgressMeters,
                    requiredTouchdowns:
                        configuration.minimumTargetTouchdowns,
                    requiredAlternatingSteps:
                        configuration.minimumTargetAlternatingSteps,
                    requiredSwingFootLiftMeters:
                        configuration.minimumTargetSwingFootLiftMeters,
                    objectiveSwingFootLiftMeters:
                        configuration
                            .targetDiscoveryObjectiveSwingFootLiftMeters,
                    requiredFootAirTimeSeconds:
                        configuration.minimumTargetFootAirTimeSeconds,
                    objectiveFootAirTimeSeconds:
                        configuration
                            .targetDiscoveryObjectiveFootAirTimeSeconds,
                    requiredFootUnloadingFraction:
                        configuration.minimumTargetFootUnloadingFraction,
                    objectiveFootUnloadingFraction:
                        configuration
                            .targetDiscoveryObjectiveFootUnloadingFraction,
                    requiredTerminalFootUnloadingFraction:
                        configuration
                            .minimumTargetTerminalFootUnloadingFraction,
                    requiredClearanceMeters:
                        configuration.minimumTargetClearanceMeters,
                    requiredGraspQuality:
                        configuration.minimumTargetGraspQuality,
                    requiredFeasibilityDwellSteps:
                        configuration.targetFeasibilityDwellSteps)
            }
            return best
        }

        func targetSelectionLoss(_ target: Target) -> Float {
            let object = target.state.manipulation.object
            let root = target.state.humanoid.root
            return PhysicalFlowBalancedObjective.evaluate(
                normalizedErrors: [
                    max(targetSelectionClearance - target.clearance, 0)
                        / max(targetSelectionClearance, 0.02),
                    length(object.linearVelocity) / 0.15,
                    length(object.angularVelocity) / 0.50,
                    length(root.linearVelocity) / 0.30,
                    length(root.angularVelocity) / 0.60,
                    max(0.9 - target.boxUpright, 0) / 0.15,
                    max(0.9 - target.robotUpright, 0) / 0.15,
                    max(configuration.minimumTargetCarryDistanceMeters
                        - target.carryDistance, 0)
                        / max(configuration.minimumTargetCarryDistanceMeters,
                              0.05),
                    targetSelectionDestinationProgress > 0
                        ? max(targetSelectionDestinationProgress
                            - target.destinationProgress, 0)
                            / max(targetSelectionDestinationProgress, 0.05)
                        : 0,
                    targetSelectionRootDestinationProgress > 0
                        ? max(targetSelectionRootDestinationProgress
                            - target.rootDestinationProgress, 0)
                            / max(
                                targetSelectionRootDestinationProgress,
                                0.05)
                        : 0,
                    targetDiscoverySwingFootLift > 0
                        ? max(targetDiscoverySwingFootLift
                            - target.maximumSwingFootLift, 0)
                            / 0.01
                        : 0,
                    targetDiscoveryFootAirTime > 0
                        ? max(targetDiscoveryFootAirTime
                            - target.maximumLoadedFootAirTime, 0)
                            / max(targetDiscoveryFootAirTime, 0.02)
                        : 0,
                    targetDiscoveryFootUnloading > 0
                        ? max(targetDiscoveryFootUnloading
                            - target.maximumFootUnloadingFraction, 0)
                            / 0.10
                        : 0,
                    max(targetDiscoveryGraspQuality
                        - target.flags.graspQuality, 0)
                        / max(targetDiscoveryGraspQuality, 0.25),
                ]).bottleneckLoss
        }

        var simulatedEnvironmentControlSteps = 0
        var targetDiscoveryCandidateRollouts = 0
        // A neutral composed-stage proposal must continue holding the
        // certified grasp throughout its horizon. Otherwise a trajectory
        // imported from the preceding stage quietly replays obsolete arm
        // knots after the boundary. Full-body discovery may still optimize
        // these seeded knots; lower-body-only discovery freezes them.
        let seedsFromSourceTerminalTrajectory =
            configuration.continueTrajectoryFromSourceTerminal
                && sourceTerminalBoundary != nil
                && !configuration
                    .preserveProvidedContinuationSeedUpperBody
        let preservesProvidedUpperBodySeed =
            configuration.continueTrajectoryFromSourceTerminal
                && sourceTerminalBoundary != nil
                && configuration
                    .preserveProvidedContinuationSeedUpperBody
        let holdsSourceTerminalUpperBody =
            seedsFromSourceTerminalTrajectory
                && configuration.targetDiscoveryLowerBodyOnly
        var selectedTargetTrajectory = seedsFromSourceTerminalTrajectory
            ? trajectoryHoldingSourceTerminalUpperBody(
                targetTrajectory,
                boundary: sourceTerminalBoundary!,
                armKnotCount: configuration.trajectoryKnotCount,
                blendKnotCount: configuration.legBlendKnotCount,
                legResidualKnotCount:
                    configuration.legResidualKnotCount,
                torsoResidualKnotCount:
                    configuration.torsoResidualKnotCount,
                armAsymmetryKnotCount:
                    configuration.armAsymmetryKnotCount,
                maximumArmAsymmetryAction:
                    configuration.maximumArmAsymmetryAction)
            : targetTrajectory

        if configuration.targetDiscoveryPopulationSize > 0
            && configuration.recedingHorizonSteps == 0 {
            let count = configuration.targetDiscoveryPopulationSize
            var generator = ProbeRandomNumberGenerator(
                seed: optimizerSeed &+ 0xD15C_0A3E)
            var mean = selectedTargetTrajectory
            var covariance = [[Float]](
                repeating: [Float](repeating: 0, count: parameterCount),
                count: parameterCount)
            for index in 0..<parameterCount {
                covariance[index][index] =
                    configuration.targetDiscoveryInitialStandardDeviation
                    * configuration.targetDiscoveryInitialStandardDeviation
            }
            var overallBest: TargetDiscoveryCandidate?

            func discoverySample(
                around center: [Float], transform: [[Float]]
            ) -> [Float] {
                let noise = center.map { _ in generator.normal() }
                return structuredTargetDiscoveryTrajectory(
                    center.indices.map { row in
                    guard targetDiscoveryParameterIsActive(row) else {
                        return center[row]
                    }
                    let delta = (0...row).reduce(Float(0)) {
                        $0 + transform[row][$1] * noise[$1]
                    }
                    return simd_clamp(center[row] + delta, -0.999, 0.999)
                })
            }

            func evaluateDiscovery(
                _ parameters: [[Float]]
            ) throws -> [TargetDiscoveryCandidate] {
                var runtime = try prepareSource(count: parameters.count)
                let graspFeedback =
                    configuration.graspAnchorFeedbackBlend > 0
                    ? captureGraspFeedback(
                        task: runtime.task,
                        blend: configuration.graspAnchorFeedbackBlend)
                    : nil
                let initialPlacementDistances = Array(
                    runtime.result.metrics["state/placement_distance_m"]!)
                let initialRootDestinationDistances = Array(
                    runtime.result.metrics[
                        "state/root_destination_distance_m"]!)
                let initialAlternatingSteps = Array(
                    runtime.result.metrics[
                        "state/loaded_alternating_steps"]!)
                let initialTouchdowns = Array(
                    runtime.result.metrics["state/loaded_touchdowns"]!)
                var maximumSwingFootLifts = [Float](
                    repeating: 0, count: parameters.count)
                var maximumLoadedFootAirTimes = [Float](
                    repeating: 0, count: parameters.count)
                var maximumFootUnloadingFractions = [Float](
                    repeating: 0, count: parameters.count)
                var bestLosses = [Float](
                    repeating: .infinity, count: parameters.count)
                var bestErrors = [Float](
                    repeating: .infinity, count: parameters.count)
                var bestSteps = [Int](repeating: 0, count: parameters.count)
                var stablePathViolationSteps = [Int](
                    repeating: 0, count: parameters.count)
                var feasibilityDwells = [Int](
                    repeating: 0, count: parameters.count)
                var feasibilityWindows = [[Float]](
                    repeating: [], count: parameters.count)
                for step in 0..<targetExecutionSteps {
                    let actions = try applyTrajectory(
                        parameters, step: step,
                        denominator: configuration.targetGenerationSteps,
                        task: runtime.task,
                        observation: runtime.observation,
                        trajectoryStart: currentTrajectoryStart,
                        graspFeedback: graspFeedback)
                    try runtime.task.step(
                        actions: actions, into: &runtime.result)
                    runtime.observation = runtime.result.observations
                    let allStates = states(runtime.task)
                    let allFlags = flags(
                        task: runtime.task,
                        observation: runtime.observation,
                        result: runtime.result)
                    let clearances = runtime.result.metrics[
                        "state/box_clearance_m"]!
                    let carryDistances = runtime.result.metrics[
                        "state/carry_distance_m"]!
                    let placementDistances = runtime.result.metrics[
                        "state/placement_distance_m"]!
                    let rootDestinationDistances = runtime.result.metrics[
                        "state/root_destination_distance_m"]!
                    let alternatingSteps = runtime.result.metrics[
                        "state/loaded_alternating_steps"]!
                    let touchdowns = runtime.result.metrics[
                        "state/loaded_touchdowns"]!
                    let loadedFootAirTimes = runtime.result.metrics[
                        "state/maximum_loaded_foot_air_time_s"]!
                    let footUnloadingFractions = runtime.result.metrics[
                        "state/foot_unloading_fraction"]!
                    let swingClearances = runtime.result.metrics[
                        "state/maximum_loaded_swing_clearance_m"]!
                    for environment in parameters.indices {
                        let destinationProgress =
                            initialPlacementDistances[environment]
                                - placementDistances[environment]
                        let rootDestinationProgress =
                            initialRootDestinationDistances[environment]
                                - rootDestinationDistances[environment]
                        let loadedAlternatingSteps = max(
                            0, Int(alternatingSteps[environment]
                                - initialAlternatingSteps[environment]))
                        let loadedTouchdowns = max(
                            0, Int(touchdowns[environment]
                                - initialTouchdowns[environment]))
                        maximumSwingFootLifts[environment] = max(
                            maximumSwingFootLifts[environment],
                            swingClearances[environment])
                        maximumLoadedFootAirTimes[environment] = max(
                            maximumLoadedFootAirTimes[environment],
                            loadedFootAirTimes[environment])
                        maximumFootUnloadingFractions[environment] = max(
                            maximumFootUnloadingFractions[environment],
                            footUnloadingFractions[environment])
                        let rootUp = allStates[environment].humanoid.root
                            .rotation.act(F3(0, 0, 1)).z
                        let boxUp = allStates[environment].manipulation.object
                            .rotation.act(F3(0, 0, 1)).z
                        let stable = allFlags[environment].loadBearingGrasp
                            && allFlags[environment].unsupported
                            && allFlags[environment].physicallyLifted
                            && rootUp > 0.9 && boxUp > 0.9
                            && clearances[environment] >= 0.01
                            && pathDownwardVelocityFeasible(
                                boxVerticalVelocity: allStates[environment]
                                    .manipulation.object.linearVelocity.z,
                                maximumDownwardVelocity: configuration
                                    .maximumTargetPathDownwardBoxVelocityMPS)
                            && !allFlags[environment].failed
                        if !stable {
                            stablePathViolationSteps[environment] += 1
                        }
                        let instantFeasible =
                            allFlags[environment].loadBearingGrasp
                            && allFlags[environment].unsupported
                            && allFlags[environment].physicallyLifted
                            && rootUp > 0.9 && boxUp > 0.9
                            && !allFlags[environment].failed
                            && carryDistances[environment] >= configuration
                                .minimumTargetCarryDistanceMeters
                            && minimumDestinationProgressPassed(
                                destinationProgress,
                                minimum: configuration
                                    .minimumTargetDestinationProgressMeters)
                            && rootDestinationProgress >= configuration
                                .minimumTargetRootDestinationProgressMeters
                            && loadedTouchdowns >= configuration
                                .minimumTargetTouchdowns
                            && loadedAlternatingSteps >= configuration
                                .minimumTargetAlternatingSteps
                            && maximumSwingFootLifts[environment]
                                >= configuration
                                    .minimumTargetSwingFootLiftMeters
                            && maximumLoadedFootAirTimes[environment]
                                >= configuration
                                    .minimumTargetFootAirTimeSeconds
                            && maximumFootUnloadingFractions[environment]
                                >= configuration
                                    .minimumTargetFootUnloadingFraction
                            && clearances[environment] >= configuration
                                .minimumTargetClearanceMeters
                            && allFlags[environment].graspQuality
                                >= configuration.minimumTargetGraspQuality
                            && pathDownwardVelocityFeasible(
                                boxVerticalVelocity: allStates[environment]
                                    .manipulation.object.linearVelocity.z,
                                maximumDownwardVelocity: configuration
                                    .maximumTargetPathDownwardBoxVelocityMPS)
                            && (!configuration.requireStableCarryPath
                                || stablePathViolationSteps[environment] == 0)
                        feasibilityDwells[environment] = instantFeasible
                            ? feasibilityDwells[environment] + 1 : 0
                        let instantWindowPenalty = [
                            hardMinimumPenalty(
                                value: allFlags[environment]
                                    .graspFrictionSupportFraction,
                                minimum: 1, scale: 0.25),
                            allFlags[environment].unsupported ? 0 : 10,
                            allFlags[environment].physicallyLifted ? 0 : 10,
                            rootUp > 0.9 ? 0 : 10,
                            boxUp > 0.9 ? 0 : 10,
                            allFlags[environment].failed ? 20 : 0,
                            hardMinimumPenalty(
                                value: carryDistances[environment],
                                minimum: configuration
                                    .minimumTargetCarryDistanceMeters,
                                scale: 0.01),
                            configuration
                                .minimumTargetDestinationProgressMeters > 0
                                ? hardMinimumPenalty(
                                    value: destinationProgress,
                                    minimum: configuration
                                        .minimumTargetDestinationProgressMeters,
                                    scale: 0.01)
                                : 0,
                            configuration
                                .minimumTargetRootDestinationProgressMeters > 0
                                ? hardMinimumPenalty(
                                    value: rootDestinationProgress,
                                    minimum: configuration
                                        .minimumTargetRootDestinationProgressMeters,
                                    scale: 0.01)
                                : 0,
                            hardIntegerMinimumPenalty(
                                value: loadedTouchdowns,
                                minimum: configuration
                                    .minimumTargetTouchdowns),
                            hardIntegerMinimumPenalty(
                                value: loadedAlternatingSteps,
                                minimum: configuration
                                    .minimumTargetAlternatingSteps),
                            hardMinimumPenalty(
                                value: maximumSwingFootLifts[environment],
                                minimum: configuration
                                    .minimumTargetSwingFootLiftMeters,
                                scale: 0.015),
                            hardMinimumPenalty(
                                value:
                                    maximumLoadedFootAirTimes[environment],
                                minimum: configuration
                                    .minimumTargetFootAirTimeSeconds,
                                scale: 0.02),
                            hardMinimumPenalty(
                                value: maximumFootUnloadingFractions[
                                    environment],
                                minimum: configuration
                                    .minimumTargetFootUnloadingFraction,
                                scale: 0.10),
                            hardMinimumPenalty(
                                value: clearances[environment],
                                minimum: configuration
                                    .minimumTargetClearanceMeters,
                                scale: 0.01),
                            hardMinimumPenalty(
                                value: allFlags[environment].graspQuality,
                                minimum: configuration
                                    .minimumTargetGraspQuality,
                                scale: 0.05),
                            pathDownwardVelocityPenalty(
                                boxVerticalVelocity: allStates[environment]
                                    .manipulation.object.linearVelocity.z,
                                maximumDownwardVelocity: configuration
                                    .maximumTargetPathDownwardBoxVelocityMPS),
                        ].max()!
                        appendFeasibilityPenalty(
                            instantWindowPenalty,
                            to: &feasibilityWindows[environment],
                            required:
                                configuration.targetFeasibilityDwellSteps)
                        let evaluation = frontierEvaluation(
                            state: allStates[environment],
                            flags: allFlags[environment],
                            clearance: clearances[environment],
                            carryDistance: carryDistances[environment],
                            minimumCarryDistance: configuration
                                .targetDiscoveryObjectiveCarryDistanceMeters
                                ?? configuration
                                    .minimumTargetCarryDistanceMeters,
                            destinationProgress: destinationProgress,
                            minimumDestinationProgress: configuration
                                .minimumTargetDestinationProgressMeters,
                            objectiveDestinationProgress:
                                targetSelectionDestinationProgress,
                            rootDestinationProgress:
                                rootDestinationProgress,
                            minimumRootDestinationProgress: configuration
                                .minimumTargetRootDestinationProgressMeters,
                            objectiveRootDestinationProgress:
                                targetSelectionRootDestinationProgress,
                            loadedTouchdowns:
                                loadedTouchdowns,
                            minimumTouchdowns:
                                configuration.minimumTargetTouchdowns,
                            loadedAlternatingSteps:
                                loadedAlternatingSteps,
                            minimumAlternatingSteps:
                                configuration.minimumTargetAlternatingSteps,
                            maximumSwingFootLift:
                                maximumSwingFootLifts[environment],
                            minimumSwingFootLift: configuration
                                .minimumTargetSwingFootLiftMeters,
                            objectiveSwingFootLift:
                                targetDiscoverySwingFootLift,
                            maximumFootAirTime:
                                maximumLoadedFootAirTimes[environment],
                            minimumFootAirTime: configuration
                                .minimumTargetFootAirTimeSeconds,
                            objectiveFootAirTime:
                                targetDiscoveryFootAirTime,
                            maximumFootUnloading:
                                maximumFootUnloadingFractions[environment],
                            minimumFootUnloading: configuration
                                .minimumTargetFootUnloadingFraction,
                            objectiveFootUnloading:
                                targetDiscoveryFootUnloading,
                            minimumClearance: configuration
                                .minimumTargetClearanceMeters,
                            objectiveClearance: targetDiscoveryClearance,
                            minimumGraspQuality:
                                configuration.minimumTargetGraspQuality,
                            objectiveGraspQuality:
                                targetDiscoveryGraspQuality,
                            feasibilityDwellSteps:
                                feasibilityDwells[environment],
                            requiredDwellSteps:
                                configuration.targetFeasibilityDwellSteps,
                            feasibilityWindowPenalty:
                                feasibilityWindowPenalty(
                                    feasibilityWindows[environment],
                                    required: configuration
                                        .targetFeasibilityDwellSteps),
                            stablePathViolationSteps:
                                stablePathViolationSteps[environment],
                            trajectorySteps: step + 1,
                            requireStableCarryPath:
                                configuration.requireStableCarryPath)
                        if evaluation.bottleneckLoss
                            < bestLosses[environment] {
                            bestLosses[environment] =
                                evaluation.bottleneckLoss
                            bestErrors[environment] =
                                evaluation.maximumNormalizedError
                            bestSteps[environment] = step + 1
                        }
                    }
                }
                simulatedEnvironmentControlSteps += parameters.count
                        * (runtime.simulatedPreparationSteps
                            + targetExecutionSteps)
                targetDiscoveryCandidateRollouts += parameters.count
                return parameters.indices.map {
                    TargetDiscoveryCandidate(
                        parameters: parameters[$0],
                        loss: bestLosses[$0],
                        maximumNormalizedError: bestErrors[$0],
                        bestStep: bestSteps[$0])
                }
            }

            for _ in 0..<configuration.targetDiscoveryGenerations {
                let transform = cholesky(covariance)
                var parameters = [[Float]]()
                parameters.reserveCapacity(count)
                for index in 0..<count {
                    if index == 0 {
                        parameters.append(mean)
                    } else if index == 1, let overallBest {
                        parameters.append(overallBest.parameters)
                    } else {
                        parameters.append(discoverySample(
                            around: mean, transform: transform))
                    }
                }
                let candidates = try evaluateDiscovery(parameters).sorted {
                    $0.loss < $1.loss
                }
                guard let generationBest = candidates.first,
                      generationBest.loss.isFinite else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite humanoid-box target discovery")
                }
                if overallBest == nil
                    || generationBest.loss < overallBest!.loss {
                    overallBest = generationBest
                }
                let eliteCount = max(
                    2, Int(Float(count) * configuration.eliteFraction))
                let elites = candidates.prefix(eliteCount)
                var nextMean = [Float](
                    repeating: 0, count: parameterCount)
                for elite in elites {
                    for index in 0..<parameterCount {
                        nextMean[index] += elite.parameters[index]
                    }
                }
                for index in 0..<parameterCount {
                    nextMean[index] /= Float(eliteCount)
                }
                var nextCovariance = [[Float]](
                    repeating: [Float](
                        repeating: 0, count: parameterCount),
                    count: parameterCount)
                for elite in elites {
                    for row in 0..<parameterCount {
                        for column in 0..<parameterCount {
                            nextCovariance[row][column] +=
                                (elite.parameters[row] - nextMean[row])
                                * (elite.parameters[column]
                                    - nextMean[column])
                        }
                    }
                }
                for row in 0..<parameterCount {
                    for column in 0..<parameterCount {
                        nextCovariance[row][column] /= Float(eliteCount)
                    }
                    nextCovariance[row][row] = max(
                        nextCovariance[row][row], 0.02 * 0.02)
                }
                mean = zip(mean, nextMean).map {
                    0.25 * $0.0 + 0.75 * $0.1
                }
                for row in 0..<parameterCount {
                    for column in 0..<parameterCount {
                        covariance[row][column] = 0.25
                            * covariance[row][column]
                            + 0.75 * nextCovariance[row][column]
                    }
                }
            }
            guard let discovered = overallBest else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-box target discovery produced no candidate")
            }
            selectedTargetTrajectory = discovered.parameters
        }

        func generateRecedingTarget(
            using seedTrajectory: [Float]
        ) throws -> Target {
            let count = configuration.targetDiscoveryPopulationSize
            let horizon = configuration.recedingHorizonSteps
            precondition(count >= 8 && horizon > 0)
            let validationReplicaCount =
                configuration.recedingValidationReplicaCount ?? count
            precondition((1...count).contains(validationReplicaCount))
            let noiseReplicaCount = configuration
                    .optimizationActionNoiseStandardDeviation > 0
                ? configuration.optimizationActionNoiseReplicaCount : 1
            let runtimeCount = count * noiseReplicaCount
            var runtime = try prepareSource(count: runtimeCount)
            // The derived stage is serialized with
            // `canonicalizeReplicasBeforeExecution`. Search must begin from
            // that same row-zero branch boundary; otherwise contact-rich
            // source replay can leave validation rows with different hidden
            // warm starts and optimize a transition that future exact replay
            // never executes.
            canonicalizeSpeculationRows(
                runtime: &runtime, count: runtimeCount)
            let graspFeedback = configuration.graspAnchorFeedbackBlend > 0
                ? captureGraspFeedback(
                    task: runtime.task,
                    blend: configuration.graspAnchorFeedbackBlend)
                : nil
            let archivedTargetBoundaryReplay =
                archivedTargetTrajectorySequence.map {
                    Self.structuredTrajectorySequenceBoundaryReplay(
                        $0,
                        phaseSteps:
                            archivedTargetTrajectorySequencePhaseSteps,
                        denominator: horizon,
                        initial: currentTrajectoryStart,
                        armKnotCount: configuration.trajectoryKnotCount,
                        blendKnotCount: configuration.legBlendKnotCount,
                        legResidualKnotCount:
                            configuration.legResidualKnotCount,
                        maximumLegResidualAction:
                            configuration.maximumLegResidualAction,
                        torsoResidualKnotCount:
                            configuration.torsoResidualKnotCount,
                        maximumTorsoResidualAction:
                            configuration.maximumTorsoResidualAction,
                        armAsymmetryKnotCount:
                            configuration.armAsymmetryKnotCount,
                        maximumArmAsymmetryAction:
                            configuration.maximumArmAsymmetryAction)
                }
            let initialPlacementDistances = Array(
                runtime.result.metrics["state/placement_distance_m"]!)
            let initialRootDestinationDistances = Array(
                runtime.result.metrics[
                    "state/root_destination_distance_m"]!)
            let initialAlternatingSteps = Array(
                runtime.result.metrics[
                    "state/loaded_alternating_steps"]!)
            let initialTouchdowns = Array(
                runtime.result.metrics["state/loaded_touchdowns"]!)
            let initialFootUnloadingFractions = Array(
                runtime.result.metrics["state/foot_unloading_fraction"]!)
            simulatedEnvironmentControlSteps += runtimeCount
                * runtime.simulatedPreparationSteps
            runtime.simulatedPreparationSteps = 0
            var generator = ProbeRandomNumberGenerator(
                seed: optimizerSeed &+ 0x4D50_4301)
            func shiftedWarmStart(
                _ trajectory: [Float], controls: Int = 1
            ) -> [Float] {
                precondition(controls >= 0)
                var shifted = trajectory
                for _ in 0..<controls {
                    shifted = shiftedRecedingTrajectory(
                        shifted, horizon: horizon,
                        armKnotCount: configuration.trajectoryKnotCount,
                        blendKnotCount: configuration.legBlendKnotCount,
                        legResidualKnotCount:
                            configuration.legResidualKnotCount,
                        maximumLegResidualAction:
                            configuration.maximumLegResidualAction,
                        torsoResidualKnotCount:
                            configuration.torsoResidualKnotCount,
                        maximumTorsoResidualAction:
                            configuration.maximumTorsoResidualAction,
                        armAsymmetryKnotCount:
                            configuration.armAsymmetryKnotCount,
                        maximumArmAsymmetryAction:
                            configuration.maximumArmAsymmetryAction)
                }
                return shifted
            }
            var mean = configuration.recedingShiftInitialSeed
                ? shiftedWarmStart(seedTrajectory) : seedTrajectory
            var selectedSequence = [[Float]]()
            selectedSequence.reserveCapacity(targetExecutionSteps)
            var selectedSequencePhaseSteps = [Int]()
            selectedSequencePhaseSteps.reserveCapacity(targetExecutionSteps)
            var recedingTrajectoryStart = currentTrajectoryStart
            var fallbackPlan: (
                parameters: [Float],
                phaseStep: Int,
                trajectoryStart: StructuredTrajectoryBoundary?
            )?
            var remainingControlHorizonSteps = 0
            var committedDwell = 0
            var maximumCommittedDwell = 0
            var stablePath = true
            var firstPathViolation: Int?
            var predicateDwells = [Int](repeating: 0, count: 9)
            var maximumPredicateDwells = [Int](repeating: 0, count: 9)
            var maximumClearance: Float = -.infinity
            var maximumCarry: Float = 0
            var maximumStableCarry: Float = 0
            var maximumDestinationProgress: Float = -.infinity
            var maximumRootDestinationProgress: Float = -.infinity
            var maximumLoadedAlternatingSteps = 0
            var maximumLoadedTouchdowns = 0
            var maximumSwingFootLift: Float = 0
            var maximumLoadedFootAirTime: Float = 0
            var maximumFootUnloadingFraction: Float = 0
            var maximumGraspQuality: Float = -.infinity
            var speculativeMaximumRootDestinationProgress: Float =
                -.infinity
            var speculativeMaximumLoadedTouchdowns = 0
            var speculativeMaximumLoadedAlternatingSteps = 0
            var speculativeMaximumSwingFootLift: Float = 0
            var speculativeMaximumLoadedFootAirTime: Float = 0
            var speculativeMaximumFootUnloadingFraction: Float = 0
            var speculativeMaximumFeasibilityDwell = 0
            var speculativeMaximumPredicateDwells =
                [Int](repeating: 0, count: 9)
            var predictedRecoveryPathSafe = true
            var committedTrace = [HumanoidBoxPhysicalFlowTraceSample]()
            committedTrace.reserveCapacity(targetExecutionSteps + 1)

            func pathScore(
                _ candidate: TargetDiscoveryCandidate
            ) -> TargetDiscoveryPathScore {
                TargetDiscoveryPathScore(
                    firstControlSafe: candidate.firstControlSafe,
                    commitPathSafe: candidate.commitPathSafe,
                    terminalGoalFeasible:
                        candidate.terminalGoalFeasible,
                    predictedPathSafe: candidate.predictedPathSafe,
                    firstStablePathViolationStep:
                        candidate.firstStablePathViolationStep,
                    maximumFeasibilityDwellSteps:
                        candidate.maximumFeasibilityDwellSteps ?? 0,
                    terminalRecoveryViable:
                        candidate.terminalRecoveryViable,
                    loss: candidate.loss)
            }

            func pathCandidateIsBetter(
                _ candidate: TargetDiscoveryCandidate,
                than incumbent: TargetDiscoveryCandidate
            ) -> Bool {
                Self.pathScoreIsBetter(
                    pathScore(candidate), than: pathScore(incumbent))
            }

            func currentTraceSample(
                step: Int, appliedActions: RLActionBatch? = nil
            )
                -> HumanoidBoxPhysicalFlowTraceSample {
                let state = states(runtime.task)[0]
                let currentFlags = flags(
                    task: runtime.task, observation: runtime.observation,
                    result: runtime.result)[0]
                let root = state.humanoid.root
                let box = state.manipulation.object
                let localLeftHand = box.rotation.conjugate.act(
                    state.manipulation.leftHand.position - box.position)
                let localRightHand = box.rotation.conjugate.act(
                    state.manipulation.rightHand.position - box.position)
                let graspDiagnostics = graspFeedback.map {
                    graspFeedbackArmDiagnostics(
                        task: runtime.task, feedback: $0)[0]
                }
                let leftGraspDiagnostic = graspDiagnostics?[0]
                let rightGraspDiagnostic = graspDiagnostics?[1]
                let faceOffset = 0.5
                    * HumanoidBoxCarryTask.boxDimensions.y
                    + HumanoidBoxCarryTask.handCollisionSphereRadius
                let leftTarget = F3(0, faceOffset, 0)
                let rightTarget = F3(0, -faceOffset, 0)
                let handOpposition = -simd_dot(
                    simd_normalize(localLeftHand),
                    simd_normalize(localRightHand))
                let placement = runtime.result.metrics[
                    "state/placement_distance_m"]![0]
                let rootDestinationProgress =
                    initialRootDestinationDistances[0]
                        - runtime.result.metrics[
                            "state/root_destination_distance_m"]![0]
                let loadedAlternatingSteps = max(
                    0, Int(runtime.result.metrics[
                        "state/loaded_alternating_steps"]![0]
                        - initialAlternatingSteps[0]))
                let loadedTouchdowns = max(
                    0, Int(runtime.result.metrics[
                        "state/loaded_touchdowns"]![0]
                        - initialTouchdowns[0]))
                let swingFootLift = runtime.result.metrics[
                    "state/maximum_loaded_swing_clearance_m"]![0]
                let loadedFootAirTime = runtime.result.metrics[
                    "state/maximum_loaded_foot_air_time_s"]![0]
                let footUnloadingFraction = runtime.result.metrics[
                    "state/foot_unloading_fraction"]![0]
                let leftFeedbackCorrection = leftGraspDiagnostic.map {
                    diagnostic in
                    diagnostic.normalizedActionCorrection.map {
                        graspFeedback!.blend * $0
                    }
                }
                let rightFeedbackCorrection = rightGraspDiagnostic.map {
                    diagnostic in
                    diagnostic.normalizedActionCorrection.map {
                        graspFeedback!.blend * $0
                    }
                }
                let leftAppliedActions = appliedActions.map { actions in
                    Array(actions.values[
                        firstArmAction..<(firstArmAction + 4)])
                }
                let rightAppliedActions = appliedActions.map { actions in
                    let start = firstArmAction + 4
                    let end = firstArmAction + armActionCount
                    return Array(actions.values[start..<end])
                }
                let leftCompositionClamped = leftAppliedActions?.contains {
                    abs($0) >= 0.998
                } ?? false
                let rightCompositionClamped = rightAppliedActions?.contains {
                    abs($0) >= 0.998
                } ?? false
                let leftPreFeedbackActions = zip(
                    leftAppliedActions ?? [], leftFeedbackCorrection ?? [])
                    .map { $0.0 - $0.1 }
                let rightPreFeedbackActions = zip(
                    rightAppliedActions ?? [], rightFeedbackCorrection ?? [])
                    .map { $0.0 - $0.1 }
                return HumanoidBoxPhysicalFlowTraceSample(
                    step: step,
                    rootPositionMeters: [
                        root.position.x, root.position.y, root.position.z,
                    ],
                    rootLinearVelocityMPS: [
                        root.linearVelocity.x, root.linearVelocity.y,
                        root.linearVelocity.z,
                    ],
                    rootUprightAlignment: root.rotation
                        .act(F3(0, 0, 1)).z,
                    boxPositionMeters: [
                        box.position.x, box.position.y, box.position.z,
                    ],
                    boxLinearVelocityMPS: [
                        box.linearVelocity.x, box.linearVelocity.y,
                        box.linearVelocity.z,
                    ],
                    boxUprightAlignment: box.rotation
                        .act(F3(0, 0, 1)).z,
                    boxClearanceMeters: runtime.result.metrics[
                        "state/box_clearance_m"]![0],
                    carryDistanceMeters: runtime.result.metrics[
                        "state/carry_distance_m"]![0],
                    placementDistanceMeters: placement,
                    destinationProgressMeters:
                        initialPlacementDistances[0] - placement,
                    rootDestinationProgressMeters:
                        rootDestinationProgress,
                    loadedTouchdowns:
                        loadedTouchdowns,
                    loadedAlternatingSteps:
                        loadedAlternatingSteps,
                    maximumSwingFootLiftMeters: max(
                        maximumSwingFootLift, swingFootLift),
                    maximumLoadedFootAirTimeSeconds: max(
                        maximumLoadedFootAirTime, loadedFootAirTime),
                    maximumFootUnloadingFraction: max(
                        maximumFootUnloadingFraction,
                        footUnloadingFraction),
                    footUnloadingFraction: footUnloadingFraction,
                    leftFootContact: runtime.result.metrics[
                        "state/left_foot_contact"]![0] > 0.5,
                    rightFootContact: runtime.result.metrics[
                        "state/right_foot_contact"]![0] > 0.5,
                    leftLoadBearingFootContact: runtime.result.metrics[
                        "state/left_load_bearing_foot_contact"]![0] > 0.5,
                    rightLoadBearingFootContact: runtime.result.metrics[
                        "state/right_load_bearing_foot_contact"]![0] > 0.5,
                    leftFootNormalLoad: runtime.result.metrics[
                        "state/left_foot_normal_load"]![0],
                    rightFootNormalLoad: runtime.result.metrics[
                        "state/right_foot_normal_load"]![0],
                    leftFootGroundClearanceMeters: runtime.result.metrics[
                        "state/left_foot_ground_clearance_m"]![0],
                    rightFootGroundClearanceMeters: runtime.result.metrics[
                        "state/right_foot_ground_clearance_m"]![0],
                    leftLoadedFootAirTimeSeconds: runtime.result.metrics[
                        "state/left_loaded_foot_air_time_s"]![0],
                    rightLoadedFootAirTimeSeconds: runtime.result.metrics[
                        "state/right_loaded_foot_air_time_s"]![0],
                    maximumActuatorTorqueRatio: runtime.result.metrics[
                        "state/maximum_actuator_torque_ratio"]![0],
                    maximumArmActuatorTorqueRatio: runtime.result.metrics[
                        "state/maximum_arm_actuator_torque_ratio"]![0],
                    saturatedActuatorCount: Int(runtime.result.metrics[
                        "state/saturated_actuator_count"]![0]),
                    saturatedArmActuatorCount: Int(runtime.result.metrics[
                        "state/saturated_arm_actuator_count"]![0]),
                    minimumJointLimitMarginRadians: runtime.result.metrics[
                        "state/minimum_joint_limit_margin_rad"]![0],
                    maximumRequestedTargetClampRadians:
                        runtime.result.metrics[
                            "state/maximum_requested_target_clamp_rad"]![0],
                    boxRootRelativeSpeedMPS: length(
                        box.linearVelocity - root.linearVelocity),
                    leftHandContact: runtime.result.metrics[
                        "state/left_hand_contact"]![0] > 0.5,
                    rightHandContact: runtime.result.metrics[
                        "state/right_hand_contact"]![0] > 0.5,
                    leftHandNormalLoad: runtime.result.metrics[
                        "state/left_hand_normal_load"]![0],
                    rightHandNormalLoad: runtime.result.metrics[
                        "state/right_hand_normal_load"]![0],
                    graspFrictionSupportFraction: runtime.result.metrics[
                        "state/grasp_friction_support_fraction"]![0],
                    leftHandRelativeToBoxMeters: [
                        localLeftHand.x, localLeftHand.y, localLeftHand.z,
                    ],
                    rightHandRelativeToBoxMeters: [
                        localRightHand.x, localRightHand.y,
                        localRightHand.z,
                    ],
                    leftHandLinearVelocityMPS: [
                        state.manipulation.leftHand.linearVelocity.x,
                        state.manipulation.leftHand.linearVelocity.y,
                        state.manipulation.leftHand.linearVelocity.z,
                    ],
                    rightHandLinearVelocityMPS: [
                        state.manipulation.rightHand.linearVelocity.x,
                        state.manipulation.rightHand.linearVelocity.y,
                        state.manipulation.rightHand.linearVelocity.z,
                    ],
                    leftHandBoxRelativeSpeedMPS: length(
                        state.manipulation.leftHand.linearVelocity
                            - box.linearVelocity),
                    rightHandBoxRelativeSpeedMPS: length(
                        state.manipulation.rightHand.linearVelocity
                            - box.linearVelocity),
                    leftGraspFeedbackTaskDeltaMeters:
                        leftGraspDiagnostic.map {
                            [$0.taskDelta.x, $0.taskDelta.y,
                             $0.taskDelta.z]
                        },
                    rightGraspFeedbackTaskDeltaMeters:
                        rightGraspDiagnostic.map {
                            [$0.taskDelta.x, $0.taskDelta.y,
                             $0.taskDelta.z]
                        },
                    leftGraspFeedbackUnclampedTaskDeltaMagnitudeMeters:
                        leftGraspDiagnostic?
                            .unclampedTaskDeltaMagnitude,
                    rightGraspFeedbackUnclampedTaskDeltaMagnitudeMeters:
                        rightGraspDiagnostic?
                            .unclampedTaskDeltaMagnitude,
                    leftGraspFeedbackJointDeltaRadians:
                        leftGraspDiagnostic?.jointDeltaRadians,
                    rightGraspFeedbackJointDeltaRadians:
                        rightGraspDiagnostic?.jointDeltaRadians,
                    leftGraspFeedbackActionCorrection:
                        leftFeedbackCorrection,
                    rightGraspFeedbackActionCorrection:
                        rightFeedbackCorrection,
                    leftAppliedArmActions: leftAppliedActions,
                    rightAppliedArmActions: rightAppliedActions,
                    leftInferredPreFeedbackArmActions:
                        leftPreFeedbackActions.isEmpty
                            || leftCompositionClamped
                            ? nil : leftPreFeedbackActions,
                    rightInferredPreFeedbackArmActions:
                        rightPreFeedbackActions.isEmpty
                            || rightCompositionClamped
                            ? nil : rightPreFeedbackActions,
                    leftArmCompositionClamped:
                        appliedActions == nil ? nil : leftCompositionClamped,
                    rightArmCompositionClamped:
                        appliedActions == nil ? nil : rightCompositionClamped,
                    appliedNormalizedActions: appliedActions.map { actions in
                        Array(actions.values[0..<runtime.task.spec.action
                            .elementCount])
                    },
                    leftHandTargetDistanceMeters:
                        length(localLeftHand - leftTarget),
                    rightHandTargetDistanceMeters:
                        length(localRightHand - rightTarget),
                    handOppositionAlignment: handOpposition,
                    graspQuality: runtime.result.metrics[
                        "state/grasp_quality"]![0],
                    loadBearingGrasp: runtime.result.metrics[
                        "state/friction_load_bearing_grasp"]![0] > 0.5,
                    boxPedestalContact: runtime.result.metrics[
                        "state/box_pedestal_contact"]![0] > 0.5,
                    boxDestinationContact: runtime.result.metrics[
                        "state/box_destination_contact"]![0] > 0.5,
                    boxGroundContact: runtime.result.metrics[
                        "state/box_ground_contact"]![0] > 0.5,
                    unsupported: currentFlags.unsupported,
                    physicallyLifted: currentFlags.physicallyLifted,
                    failed: currentFlags.failed)
            }
            committedTrace.append(currentTraceSample(step: 0))

            func locomotionProposal(from center: [Float]) -> [Float] {
                guard let blend = configuration
                        .recedingLocomotionBlendProposal else { return center }
                return trajectoryWithLocomotionProposal(
                    center,
                    blend: blend,
                    armKnotCount: configuration.trajectoryKnotCount,
                    blendKnotCount: configuration.legBlendKnotCount,
                    legResidualKnotCount:
                        configuration.legResidualKnotCount,
                    torsoResidualKnotCount:
                        configuration.torsoResidualKnotCount,
                    armAsymmetryKnotCount:
                        configuration.armAsymmetryKnotCount,
                    zeroResiduals: configuration
                        .recedingLocomotionZeroResidualProposal)
            }

            func sample(
                around center: [Float], transform: [[Float]],
                block: TargetDiscoverySearchBlock
            ) -> [Float] {
                let noise = center.map { _ in generator.normal() }
                return structuredTargetDiscoveryTrajectory(
                    center.indices.map { row in
                    guard targetDiscoveryParameterIsActive(
                        row, block: block) else {
                        return center[row]
                    }
                    let delta = (0...row).reduce(Float(0)) {
                        $0 + transform[row][$1] * noise[$1]
                    }
                    return simd_clamp(center[row] + delta, -0.999, 0.999)
                })
            }

            func sampleIndependent(
                around center: [Float], standardDeviation: Float,
                block: TargetDiscoverySearchBlock
            ) -> [Float] {
                precondition(
                    standardDeviation.isFinite && standardDeviation > 0)
                return structuredTargetDiscoveryTrajectory(
                    center.indices.map { index in
                        guard targetDiscoveryParameterIsActive(
                            index, block: block) else {
                            return center[index]
                        }
                        return simd_clamp(
                            center[index]
                                + standardDeviation * generator.normal(),
                            -0.999, 0.999)
                    })
            }

            for committedStep in 0..<targetExecutionSteps {
                let reusingCertifiedPlan =
                    remainingControlHorizonSteps > 0
                        && fallbackPlan != nil
                let searchPhaseStep = committedStep == 0
                    ? configuration.recedingInitialPhaseStep : 0
                let snapshot = runtime.task.captureSpeculationSnapshot()
                let committedObservation = runtime.observation
                let committedResult = runtime.result
                var covariance = [[Float]](
                    repeating: [Float](repeating: 0,
                                       count: parameterCount),
                    count: parameterCount)
                for index in 0..<parameterCount {
                    covariance[index][index] = configuration
                        .targetDiscoveryInitialStandardDeviation
                        * configuration
                            .targetDiscoveryInitialStandardDeviation
                }
                var overallBest: TargetDiscoveryCandidate?
                var overallNearMiss: TargetDiscoveryCandidate?
                var bestStableSwingCandidate:
                    TargetDiscoveryCandidate?
                var bestSwingFrontierCandidate:
                    TargetDiscoveryCandidate?
                var bestFeasibilityFrontierCandidate:
                    TargetDiscoveryCandidate?
                if let archivedTargetTrajectorySequence,
                   let archivedTargetBoundaryReplay {
                    let phase =
                        archivedTargetTrajectorySequencePhaseSteps?[
                            committedStep] ?? 0
                    overallBest = TargetDiscoveryCandidate(
                        parameters:
                            archivedTargetTrajectorySequence[committedStep],
                        loss: 0,
                        maximumNormalizedError: 0,
                        bestStep: phase + 1,
                        trajectoryStart:
                            archivedTargetBoundaryReplay.starts[
                                committedStep],
                        predictedPathSafe:
                            archivedTargetPredictedRecoveryPathSafe ?? true,
                        phaseStep: phase)
                }

                func isBetterStableSwing(
                    _ candidate: TargetDiscoveryCandidate,
                    than incumbent: TargetDiscoveryCandidate?
                ) -> Bool {
                    guard candidate.firstControlSafe,
                          candidate.predictedPathSafe,
                          (candidate.maximumLoadedFootAirTimeSeconds ?? 0)
                            >= configuration
                                .minimumTargetFootAirTimeSeconds,
                          (candidate.maximumSwingFootLiftMeters ?? 0)
                            >= configuration
                                .minimumTargetSwingFootLiftMeters else {
                        return false
                    }
                    guard let incumbent else { return true }
                    if candidate.terminalGoalFeasible
                            != incumbent.terminalGoalFeasible {
                        return candidate.terminalGoalFeasible
                    }
                    if candidate.terminalRecoveryViable
                            != incumbent.terminalRecoveryViable {
                        return candidate.terminalRecoveryViable
                    }
                    let candidateDwell =
                        candidate.maximumFeasibilityDwellSteps ?? 0
                    let incumbentDwell =
                        incumbent.maximumFeasibilityDwellSteps ?? 0
                    if candidateDwell != incumbentDwell {
                        return candidateDwell > incumbentDwell
                    }
                    let candidateAir =
                        candidate.maximumLoadedFootAirTimeSeconds ?? 0
                    let incumbentAir =
                        incumbent.maximumLoadedFootAirTimeSeconds ?? 0
                    if candidateAir != incumbentAir {
                        return candidateAir > incumbentAir
                    }
                    let candidateLift =
                        candidate.maximumSwingFootLiftMeters ?? 0
                    let incumbentLift =
                        incumbent.maximumSwingFootLiftMeters ?? 0
                    if candidateLift != incumbentLift {
                        return candidateLift > incumbentLift
                    }
                    return candidate.loss < incumbent.loss
                }

                func terminalRecoveryMargin(
                    _ candidate: TargetDiscoveryCandidate
                ) -> Float {
                    min(
                        min(
                            (candidate.finalClearanceMeters ?? -.infinity)
                                / configuration.minimumTargetClearanceMeters,
                            configuration.minimumTargetGraspQuality > 0
                                ? (candidate.finalGraspQuality ?? -.infinity)
                                    / configuration
                                        .minimumTargetGraspQuality
                                : 1),
                        configuration
                            .maximumTargetTerminalDownwardBoxVelocityMPS
                            .map {
                                ((candidate.finalBoxVerticalVelocityMPS
                                    ?? -.infinity) + $0)
                                    / max($0, 1e-6)
                            } ?? 1)
                }

                func swingFrontierScore(
                    _ candidate: TargetDiscoveryCandidate
                ) -> TargetDiscoverySwingFrontierScore {
                    TargetDiscoverySwingFrontierScore(
                        firstControlSafe: candidate.firstControlSafe,
                        swingMilestonePassed:
                            (candidate.maximumLoadedFootAirTimeSeconds ?? 0)
                                >= configuration
                                    .minimumTargetFootAirTimeSeconds
                                && (candidate
                                    .maximumSwingFootLiftMeters ?? 0)
                                    >= configuration
                                        .minimumTargetSwingFootLiftMeters,
                        predictedPathSafe:
                            candidate.predictedPathSafe,
                        terminalGoalFeasible:
                            candidate.terminalGoalFeasible,
                        maximumFootAirTimeSeconds:
                            candidate.maximumLoadedFootAirTimeSeconds ?? 0,
                        maximumSwingFootLiftMeters:
                            candidate.maximumSwingFootLiftMeters ?? 0,
                        firstStablePathViolationStep:
                            candidate.firstStablePathViolationStep,
                        terminalRecoveryViable:
                            candidate.terminalRecoveryViable,
                        terminalRecoveryMargin:
                            terminalRecoveryMargin(candidate),
                        graspQualityDwellSteps:
                            candidate.maximumPredicateDwellSteps?[7] ?? 0,
                        loss: candidate.loss)
                }

                func feasibilityFrontierScore(
                    _ candidate: TargetDiscoveryCandidate
                ) -> TargetDiscoveryFeasibilityFrontierScore {
                    TargetDiscoveryFeasibilityFrontierScore(
                        firstControlSafe: candidate.firstControlSafe,
                        predictedPathSafe: candidate.predictedPathSafe,
                        maximumFeasibilityDwellSteps:
                            candidate.maximumFeasibilityDwellSteps ?? 0,
                        terminalGoalFeasible:
                            candidate.terminalGoalFeasible,
                        terminalRecoveryViable:
                            candidate.terminalRecoveryViable,
                        terminalRecoveryMargin:
                            terminalRecoveryMargin(candidate),
                        loss: candidate.loss)
                }

                func restoreCommittedState() {
                    runtime.task.restoreSpeculationSnapshot(snapshot)
                    runtime.observation = committedObservation
                    runtime.result = committedResult
                }

                func evaluateReceding(
                    _ parameters: [[Float]], phaseStep: Int = 0,
                    trajectoryStart: StructuredTrajectoryBoundary?
                ) throws -> [TargetDiscoveryCandidate] {
                    precondition((0..<horizon).contains(phaseStep))
                    restoreCommittedState()
                    let expandedParameters = parameters.flatMap { parameter in
                        [[Float]](
                            repeating: parameter,
                            count: noiseReplicaCount)
                    }
                    var dwells = [Int](repeating: 0,
                                       count: expandedParameters.count)
                    var maximumDwells = [Int](
                        repeating: 0, count: expandedParameters.count)
                    var violations = [Int](repeating: 0,
                                            count: expandedParameters.count)
                    var firstViolationSteps = [Int?](
                        repeating: nil, count: expandedParameters.count)
                    var predicateDwells = [[Int]](
                        repeating: [Int](repeating: 0, count: 9),
                        count: expandedParameters.count)
                    var maximumPredicateDwells = [[Int]](
                        repeating: [Int](repeating: 0, count: 9),
                        count: expandedParameters.count)
                    var maximumClearances = [Float](
                        repeating: -.infinity,
                        count: expandedParameters.count)
                    var maximumCarryDistances = [Float](
                        repeating: -.infinity,
                        count: expandedParameters.count)
                    var maximumStableCarryDistances = [Float](
                        repeating: -.infinity,
                        count: expandedParameters.count)
                    var maximumDestinationProgresses = [Float](
                        repeating: -.infinity,
                        count: expandedParameters.count)
                    var maximumRootDestinationProgresses = [Float](
                        repeating: -.infinity,
                        count: expandedParameters.count)
                    var maximumLoadedAlternatingStepCounts = [Int](
                        repeating: 0, count: expandedParameters.count)
                    var maximumLoadedTouchdownCounts = [Int](
                        repeating: 0, count: expandedParameters.count)
                    var maximumSwingFootLifts = [Float](
                        repeating: 0, count: expandedParameters.count)
                    var maximumLoadedFootAirTimes = [Float](
                        repeating: 0, count: expandedParameters.count)
                    var maximumFootUnloadingFractions = [Float](
                        repeating: 0, count: expandedParameters.count)
                    var maximumGraspQualities = [Float](
                        repeating: -.infinity,
                        count: expandedParameters.count)
                    var firstControlSafe = [Bool](
                        repeating: false, count: expandedParameters.count)
                    var firstControlTerminalPenalties = [Float](
                        repeating: 0, count: expandedParameters.count)
                    var targetEndpointFootUnloadingPassed = [Bool](
                        repeating: configuration
                            .minimumTargetTerminalFootUnloadingFraction == nil,
                        count: expandedParameters.count)
                    var targetEndpointFootUnloadingPenalties = [Float](
                        repeating: 0, count: expandedParameters.count)
                    var windows = [[Float]](
                        repeating: [], count: expandedParameters.count)
                    var noiseGenerators = expandedParameters.indices.map {
                        ProbeRandomNumberGenerator(
                            seed: optimizerSeed
                                &+ UInt64(committedStep) &* 0x9E37_79B9
                                &+ UInt64($0 % noiseReplicaCount)
                                    &* 0x85EB_CA6B
                                &+ 0xA671_7A26)
                    }
                    let activePlanSteps = horizon - phaseStep
                    let requiredSafePrefixSteps =
                        Self.recedingRequiredSafePrefixSteps(
                            activePlanSteps: activePlanSteps,
                            controlHorizonSteps:
                                configuration.recedingControlHorizonSteps,
                            safetyLookaheadSteps:
                                configuration.recedingSafetyLookaheadSteps)
                    let evaluationSteps = recedingEvaluationSteps(
                        horizon: horizon,
                        phaseStep: phaseStep,
                        terminalHoldSteps:
                            configuration.recedingTerminalHoldSteps)
                    for predictedStep in 0..<evaluationSteps {
                        // A terminal hold is a recovery rollout, not an
                        // instruction to keep applying the final swing
                        // residual. Repeating that residual makes an
                        // otherwise recoverable lift look unsafe simply
                        // because the imagined controller never replans.
                        // Roll the candidate through the optimized horizon,
                        // then hand legs and torso to the learned feedback
                        // carry controller for the configured recovery
                        // window. Preserve the terminal arm correction: it
                        // encodes the current physical grasp and discarding it
                        // can make the fallback drop an otherwise held box.
                        var actions = predictedStep < activePlanSteps
                            ? try applyTrajectory(
                                expandedParameters,
                                step: phaseStep + predictedStep,
                                denominator: horizon,
                                task: runtime.task,
                                observation: runtime.observation,
                                forwardOnlyBaseCommand: configuration
                                    .recedingForwardOnlyBaseCommand,
                                holonomicBaseCommand: configuration
                                    .recedingHolonomicBaseCommand,
                                locomotionCheckpointDirectory: configuration
                                    .recedingLocomotionCheckpointDirectory,
                                locomotionCommandSpeed: configuration
                                    .recedingLocomotionCommandSpeed,
                                trajectoryStart: trajectoryStart,
                                graspFeedback: graspFeedback)
                            : try applyTrajectory(
                                expandedParameters,
                                step: horizon - 1,
                                denominator: horizon,
                                task: runtime.task,
                                observation: runtime.observation,
                                applyLegAndTorsoTrajectory: false,
                                trajectoryStart: trajectoryStart,
                                graspFeedback: graspFeedback)
                        if configuration
                                .optimizationActionNoiseStandardDeviation > 0 {
                            for environment
                                in expandedParameters.indices {
                                let row = environment
                                    * runtime.task.spec.action.elementCount
                                for component in 0..<runtime.task.spec.action
                                        .elementCount {
                                    actions.values[row + component] =
                                        simd_clamp(
                                            actions.values[row + component]
                                                + configuration
                                                    .optimizationActionNoiseStandardDeviation
                                                    * noiseGenerators[
                                                        environment].normal(),
                                            -0.999, 0.999)
                                }
                            }
                        }
                        try runtime.task.step(
                            actions: actions, into: &runtime.result)
                        runtime.observation = runtime.result.observations
                        let predictedStates = states(runtime.task)
                        let predictedFlags = flags(
                            task: runtime.task,
                            observation: runtime.observation,
                            result: runtime.result)
                        let carry = runtime.result.metrics[
                            "state/carry_distance_m"]!
                        let clearance = runtime.result.metrics[
                            "state/box_clearance_m"]!
                        let placement = runtime.result.metrics[
                            "state/placement_distance_m"]!
                        let rootDestinationDistance = runtime.result.metrics[
                            "state/root_destination_distance_m"]!
                        let alternatingSteps = runtime.result.metrics[
                            "state/loaded_alternating_steps"]!
                        let touchdowns = runtime.result.metrics[
                            "state/loaded_touchdowns"]!
                        let loadedFootAirTimes = runtime.result.metrics[
                            "state/maximum_loaded_foot_air_time_s"]!
                        let footUnloadingFractions = runtime.result.metrics[
                            "state/foot_unloading_fraction"]!
                        let swingClearances = runtime.result.metrics[
                            "state/maximum_loaded_swing_clearance_m"]!
                        for environment in expandedParameters.indices {
                            let destinationProgress =
                                initialPlacementDistances[environment]
                                    - placement[environment]
                            let rootDestinationProgress =
                                initialRootDestinationDistances[environment]
                                    - rootDestinationDistance[environment]
                            let loadedAlternatingSteps = max(
                                0, Int(alternatingSteps[environment]
                                    - initialAlternatingSteps[environment]))
                            let loadedTouchdowns = max(
                                0, Int(touchdowns[environment]
                                    - initialTouchdowns[environment]))
                            let swingFootLift =
                                swingClearances[environment]
                            let rootUp = predictedStates[environment]
                                .humanoid.root.rotation
                                .act(F3(0, 0, 1)).z
                            let boxUp = predictedStates[environment]
                                .manipulation.object.rotation
                                .act(F3(0, 0, 1)).z
                            let stable = predictedFlags[environment]
                                    .loadBearingGrasp
                                && predictedFlags[environment].unsupported
                                && predictedFlags[environment]
                                    .physicallyLifted
                                && rootUp > 0.9 && boxUp > 0.9
                                && clearance[environment] >= 0.01
                                && pathDownwardVelocityFeasible(
                                    boxVerticalVelocity: predictedStates[
                                        environment].manipulation.object
                                        .linearVelocity.z,
                                    maximumDownwardVelocity: configuration
                                        .maximumTargetPathDownwardBoxVelocityMPS)
                                && !predictedFlags[environment].failed
                            let scheduledDestinationMinimum =
                                scheduledProgressMinimum(
                                    finalMinimum: configuration
                                        .minimumTargetDestinationProgressMeters,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let scheduledRootDestinationMinimum =
                                scheduledProgressMinimum(
                                    finalMinimum: configuration
                                        .minimumTargetRootDestinationProgressMeters,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let scheduledAlternatingStepMinimum =
                                scheduledIntegerMinimum(
                                    finalMinimum: configuration
                                        .minimumTargetAlternatingSteps,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let scheduledTouchdownMinimum =
                                scheduledIntegerMinimum(
                                    finalMinimum: configuration
                                        .minimumTargetTouchdowns,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let scheduledFootAirTimeMinimum =
                                scheduledProgressMinimum(
                                    finalMinimum: configuration
                                        .minimumTargetFootAirTimeSeconds,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let scheduledSwingFootLiftMinimum =
                                scheduledProgressMinimum(
                                    finalMinimum: configuration
                                        .minimumTargetSwingFootLiftMeters,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let scheduledFootUnloadingMinimum =
                                scheduledAbsoluteMinimum(
                                    initial: initialFootUnloadingFractions[
                                        environment],
                                    finalMinimum: configuration
                                        .minimumTargetFootUnloadingFraction,
                                    absoluteStep: committedStep
                                        + min(predictedStep + 1, horizon),
                                    executionSteps: targetExecutionSteps)
                            let predicatePasses = [
                                predictedFlags[environment].bilateral,
                                predictedFlags[environment].unsupported,
                                predictedFlags[environment].physicallyLifted,
                                rootUp > 0.9 && boxUp > 0.9
                                    && !predictedFlags[environment].failed,
                                carry[environment] >= configuration
                                    .minimumTargetCarryDistanceMeters,
                                minimumDestinationProgressPassed(
                                    destinationProgress,
                                    minimum:
                                        scheduledDestinationMinimum),
                                clearance[environment] >= configuration
                                    .minimumTargetClearanceMeters,
                                predictedFlags[environment].graspQuality
                                    >= configuration
                                        .minimumTargetGraspQuality,
                                predictedFlags[environment]
                                    .loadBearingGrasp,
                            ]
                            for predicate in predicatePasses.indices {
                                predicateDwells[environment][predicate] =
                                    predicatePasses[predicate]
                                        ? predicateDwells[
                                            environment][predicate] + 1 : 0
                                maximumPredicateDwells[
                                    environment][predicate] = max(
                                        maximumPredicateDwells[
                                            environment][predicate],
                                        predicateDwells[
                                            environment][predicate])
                            }
                            maximumClearances[environment] = max(
                                maximumClearances[environment],
                                clearance[environment])
                            maximumCarryDistances[environment] = max(
                                maximumCarryDistances[environment],
                                carry[environment])
                            if stable {
                                maximumStableCarryDistances[environment] = max(
                                    maximumStableCarryDistances[environment],
                                    carry[environment])
                            }
                            maximumDestinationProgresses[environment] = max(
                                maximumDestinationProgresses[environment],
                                destinationProgress)
                            maximumRootDestinationProgresses[environment] = max(
                                maximumRootDestinationProgresses[environment],
                                rootDestinationProgress)
                            maximumLoadedAlternatingStepCounts[environment] =
                                max(
                                    maximumLoadedAlternatingStepCounts[
                                        environment],
                                    loadedAlternatingSteps)
                            maximumLoadedTouchdownCounts[environment] = max(
                                maximumLoadedTouchdownCounts[environment],
                                loadedTouchdowns)
                            maximumSwingFootLifts[environment] = max(
                                maximumSwingFootLifts[environment],
                                swingFootLift)
                            maximumLoadedFootAirTimes[environment] = max(
                                maximumLoadedFootAirTimes[environment],
                                loadedFootAirTimes[environment])
                            maximumFootUnloadingFractions[environment] = max(
                                maximumFootUnloadingFractions[environment],
                                footUnloadingFractions[environment])
                            if Self.isRecedingPlanEndpoint(
                                predictedStep: predictedStep,
                                activePlanSteps: activePlanSteps),
                               let minimum = configuration
                                    .minimumTargetTerminalFootUnloadingFraction {
                                targetEndpointFootUnloadingPassed[
                                    environment] =
                                    footUnloadingFractions[environment]
                                        >= minimum
                                targetEndpointFootUnloadingPenalties[
                                    environment] = hardMinimumPenalty(
                                        value: footUnloadingFractions[
                                            environment],
                                        minimum: minimum,
                                        scale: 0.10)
                            }
                            maximumGraspQualities[environment] = max(
                                maximumGraspQualities[environment],
                                predictedFlags[environment].graspQuality)
                            let retentionFeasible =
                                recoveryRetentionFeasible(
                                physicalStable: stable,
                                carryDistance: carry[environment],
                                minimumCarryDistance:
                                    minimumTargetPathCarryDistance,
                                clearance: clearance[environment],
                                minimumClearance: configuration
                                    .minimumTargetClearanceMeters,
                                graspQuality:
                                    predictedFlags[environment].graspQuality,
                                minimumGraspQuality: configuration
                                    .minimumTargetGraspQuality)
                            let feasible = retentionFeasible
                                && minimumDestinationProgressPassed(
                                    destinationProgress,
                                    minimum:
                                        scheduledDestinationMinimum)
                                && rootDestinationProgress
                                    >= scheduledRootDestinationMinimum
                                && loadedTouchdowns
                                    >= scheduledTouchdownMinimum
                                && loadedAlternatingSteps
                                    >= scheduledAlternatingStepMinimum
                                && maximumSwingFootLifts[environment]
                                    >= scheduledSwingFootLiftMinimum
                                && maximumLoadedFootAirTimes[environment]
                                    >= scheduledFootAirTimeMinimum
                                && maximumFootUnloadingFractions[environment]
                                    >= scheduledFootUnloadingMinimum
                            if predictedStep == 0 {
                                // MPC normally cares about the recoverable
                                // endpoint beyond its horizon. A finite flow
                                // artifact also has a real last committed
                                // control: do not accept that final control
                                // below the configured clearance reserve or
                                // already descending too quickly merely
                                // because a later predicted hold recovers.
                                let finalCommittedControl =
                                    committedStep + 1
                                        >= targetExecutionSteps
                                let immediateTerminalViable =
                                    Self.terminalRecoveryViable(
                                        clearance: clearance[environment],
                                        minimumClearance: configuration
                                            .minimumTargetTerminalClearanceMeters,
                                        boxVerticalVelocity:
                                            predictedStates[environment]
                                                .manipulation.object
                                                .linearVelocity.z,
                                        maximumDownwardVelocity: configuration
                                            .maximumTargetTerminalDownwardBoxVelocityMPS,
                                        footUnloadingFraction:
                                            footUnloadingFractions[
                                                environment],
                                        minimumFootUnloading: configuration
                                            .minimumTargetTerminalFootUnloadingFraction)
                                let immediateTerminalPenalty =
                                    finalCommittedControl
                                    ? max(
                                        configuration
                                            .minimumTargetTerminalClearanceMeters
                                            .map {
                                                hardMinimumPenalty(
                                                    value: clearance[
                                                        environment],
                                                    minimum: $0,
                                                    scale: 0.01)
                                            } ?? 0,
                                        configuration
                                            .maximumTargetTerminalDownwardBoxVelocityMPS
                                            .map {
                                                hardMinimumPenalty(
                                                    value: predictedStates[
                                                        environment]
                                                        .manipulation.object
                                                        .linearVelocity.z,
                                                    minimum: -$0,
                                                    scale: 0.05)
                                            } ?? 0)
                                    : 0
                                firstControlTerminalPenalties[environment] =
                                    immediateTerminalPenalty
                                firstControlSafe[environment] =
                                    retentionFeasible
                                        && (!finalCommittedControl
                                            || immediateTerminalViable)
                            }
                            if !retentionFeasible {
                                violations[environment] += 1
                                if firstViolationSteps[environment] == nil {
                                    firstViolationSteps[environment] =
                                        predictedStep + 1
                                }
                            }
                            dwells[environment] = feasible
                                ? dwells[environment] + 1 : 0
                            maximumDwells[environment] = max(
                                maximumDwells[environment],
                                dwells[environment])
                            appendFeasibilityPenalty([
                                stable ? 0 : 10,
                                hardMinimumPenalty(
                                    value: carry[environment],
                                    minimum: configuration
                                        .minimumTargetCarryDistanceMeters,
                                    scale: 0.01),
                                scheduledDestinationMinimum > 0
                                    ? hardMinimumPenalty(
                                        value: destinationProgress,
                                        minimum:
                                            scheduledDestinationMinimum,
                                        scale: 0.01)
                                    : 0,
                                scheduledRootDestinationMinimum > 0
                                    ? hardMinimumPenalty(
                                        value: rootDestinationProgress,
                                        minimum:
                                            scheduledRootDestinationMinimum,
                                        scale: 0.01)
                                    : 0,
                                hardIntegerMinimumPenalty(
                                    value: loadedTouchdowns,
                                    minimum: scheduledTouchdownMinimum),
                                hardIntegerMinimumPenalty(
                                    value: loadedAlternatingSteps,
                                    minimum:
                                        scheduledAlternatingStepMinimum),
                                hardMinimumPenalty(
                                    value: maximumSwingFootLifts[
                                        environment],
                                    minimum:
                                        scheduledSwingFootLiftMinimum,
                                    scale: 0.015),
                                hardMinimumPenalty(
                                    value: maximumLoadedFootAirTimes[
                                        environment],
                                    minimum:
                                        scheduledFootAirTimeMinimum,
                                    scale: 0.02),
                                hardMinimumPenalty(
                                    value: maximumFootUnloadingFractions[
                                        environment],
                                    minimum:
                                        scheduledFootUnloadingMinimum,
                                    scale: 0.10),
                                hardMinimumPenalty(
                                    value: clearance[environment],
                                    minimum: configuration
                                        .minimumTargetClearanceMeters,
                                    scale: 0.01),
                                hardMinimumPenalty(
                                    value: predictedFlags[environment]
                                        .graspQuality,
                                    minimum: configuration
                                        .minimumTargetGraspQuality,
                                    scale: 0.05),
                                pathDownwardVelocityPenalty(
                                    boxVerticalVelocity: predictedStates[
                                        environment].manipulation.object
                                        .linearVelocity.z,
                                    maximumDownwardVelocity: configuration
                                        .maximumTargetPathDownwardBoxVelocityMPS),
                            ].max()!,
                            to: &windows[environment],
                            required:
                                configuration.targetFeasibilityDwellSteps)
                        }
                    }
                    simulatedEnvironmentControlSteps +=
                        expandedParameters.count
                        * evaluationSteps
                    targetDiscoveryCandidateRollouts +=
                        expandedParameters.count
                    let terminalStates = states(runtime.task)
                    let terminalFlags = flags(
                        task: runtime.task,
                        observation: runtime.observation,
                        result: runtime.result)
                    let carry = runtime.result.metrics[
                        "state/carry_distance_m"]!
                    let clearance = runtime.result.metrics[
                        "state/box_clearance_m"]!
                    let placement = runtime.result.metrics[
                        "state/placement_distance_m"]!
                    let terminalAbsoluteStep =
                        Self.recedingTerminalAbsoluteStep(
                            committedStep: committedStep,
                            horizon: horizon,
                            phaseStep: phaseStep)
                    let scheduledTerminalDestinationMinimum =
                        scheduledProgressMinimum(
                            finalMinimum: configuration
                                .minimumTargetDestinationProgressMeters,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    let scheduledTerminalRootDestinationMinimum =
                        scheduledProgressMinimum(
                            finalMinimum: configuration
                                .minimumTargetRootDestinationProgressMeters,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    let scheduledTerminalAlternatingStepMinimum =
                        scheduledIntegerMinimum(
                            finalMinimum: configuration
                                .minimumTargetAlternatingSteps,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    let scheduledTerminalTouchdownMinimum =
                        scheduledIntegerMinimum(
                            finalMinimum: configuration
                                .minimumTargetTouchdowns,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    let scheduledTerminalFootAirTimeMinimum =
                        scheduledProgressMinimum(
                            finalMinimum: configuration
                                .minimumTargetFootAirTimeSeconds,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    let scheduledTerminalSwingFootLiftMinimum =
                        scheduledProgressMinimum(
                            finalMinimum: configuration
                                .minimumTargetSwingFootLiftMeters,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    let scheduledTerminalFootUnloadingMinimum =
                        scheduledProgressMinimum(
                            finalMinimum: configuration
                                .minimumTargetFootUnloadingFraction,
                            absoluteStep: terminalAbsoluteStep,
                            executionSteps: targetExecutionSteps)
                    return parameters.indices.map { candidate in
                        let firstReplica = candidate * noiseReplicaCount
                        let replicaRange = firstReplica..<(firstReplica
                            + noiseReplicaCount)
                        let terminalGoalComponents: [String: Bool] = [
                            "destination_progress":
                                replicaRange.allSatisfy {
                                    minimumDestinationProgressPassed(
                                        initialPlacementDistances[$0]
                                            - placement[$0],
                                        minimum:
                                            scheduledTerminalDestinationMinimum)
                                },
                            "root_destination_progress":
                                replicaRange.allSatisfy {
                                    initialRootDestinationDistances[$0]
                                        - runtime.result.metrics[
                                            "state/root_destination_distance_m"]![
                                                $0]
                                        >= scheduledTerminalRootDestinationMinimum
                                },
                            "touchdowns": replicaRange.allSatisfy {
                                Int(runtime.result.metrics[
                                    "state/loaded_touchdowns"]![$0]
                                    - initialTouchdowns[$0])
                                    >= scheduledTerminalTouchdownMinimum
                            },
                            "alternating_steps": replicaRange.allSatisfy {
                                Int(runtime.result.metrics[
                                    "state/loaded_alternating_steps"]![$0]
                                    - initialAlternatingSteps[$0])
                                    >= scheduledTerminalAlternatingStepMinimum
                            },
                            "swing_foot_lift": replicaRange.allSatisfy {
                                maximumSwingFootLifts[$0]
                                    >= scheduledTerminalSwingFootLiftMinimum
                            },
                            "foot_air_time": replicaRange.allSatisfy {
                                maximumLoadedFootAirTimes[$0]
                                    >= scheduledTerminalFootAirTimeMinimum
                            },
                            "foot_unloading": replicaRange.allSatisfy {
                                maximumFootUnloadingFractions[$0]
                                    >= scheduledTerminalFootUnloadingMinimum
                            },
                            "terminal_foot_unloading":
                                replicaRange.allSatisfy {
                                    targetEndpointFootUnloadingPassed[$0]
                                },
                        ]
                        let robustMaximumPredicateDwells = (0..<9).map {
                            predicate in
                            replicaRange.map {
                                maximumPredicateDwells[$0][predicate]
                            }.min()!
                        }
                        let evaluations = replicaRange.map { environment in
                            let destinationProgress =
                                initialPlacementDistances[environment]
                                    - placement[environment]
                            let rootDestinationProgress =
                                initialRootDestinationDistances[environment]
                                    - runtime.result.metrics[
                                        "state/root_destination_distance_m"]![
                                            environment]
                            let loadedAlternatingSteps = max(
                                0, Int(runtime.result.metrics[
                                    "state/loaded_alternating_steps"]![
                                        environment]
                                    - initialAlternatingSteps[environment]))
                            let loadedTouchdowns = max(
                                0, Int(runtime.result.metrics[
                                    "state/loaded_touchdowns"]![environment]
                                    - initialTouchdowns[environment]))
                            return frontierEvaluation(
                                state: terminalStates[environment],
                                flags: terminalFlags[environment],
                                clearance: clearance[environment],
                                carryDistance: carry[environment],
                                minimumCarryDistance: configuration
                                    .targetDiscoveryObjectiveCarryDistanceMeters
                                    ?? configuration
                                        .minimumTargetCarryDistanceMeters,
                                destinationProgress: destinationProgress,
                                minimumDestinationProgress: configuration
                                    .minimumTargetDestinationProgressMeters > 0
                                        ? scheduledTerminalDestinationMinimum
                                        : 0,
                                objectiveDestinationProgress:
                                    targetSelectionDestinationProgress,
                                rootDestinationProgress:
                                    rootDestinationProgress,
                                minimumRootDestinationProgress:
                                    scheduledTerminalRootDestinationMinimum,
                                objectiveRootDestinationProgress:
                                    targetSelectionRootDestinationProgress,
                                loadedTouchdowns:
                                    loadedTouchdowns,
                                minimumTouchdowns:
                                    scheduledTerminalTouchdownMinimum,
                                loadedAlternatingSteps:
                                    loadedAlternatingSteps,
                                minimumAlternatingSteps:
                                    scheduledTerminalAlternatingStepMinimum,
                                maximumSwingFootLift:
                                    maximumSwingFootLifts[environment],
                                minimumSwingFootLift:
                                    scheduledTerminalSwingFootLiftMinimum,
                                objectiveSwingFootLift:
                                    targetDiscoverySwingFootLift,
                                maximumFootAirTime:
                                    maximumLoadedFootAirTimes[environment],
                                minimumFootAirTime:
                                    scheduledTerminalFootAirTimeMinimum,
                                objectiveFootAirTime:
                                    targetDiscoveryFootAirTime,
                                maximumFootUnloading:
                                    maximumFootUnloadingFractions[
                                        environment],
                                minimumFootUnloading:
                                    scheduledTerminalFootUnloadingMinimum,
                                objectiveFootUnloading:
                                    targetDiscoveryFootUnloading,
                                minimumClearance: configuration
                                    .minimumTargetClearanceMeters,
                                objectiveClearance: targetDiscoveryClearance,
                                minimumGraspQuality:
                                    configuration.minimumTargetGraspQuality,
                                objectiveGraspQuality:
                                    targetDiscoveryGraspQuality,
                                feasibilityDwellSteps: dwells[environment],
                                requiredDwellSteps: configuration
                                    .targetFeasibilityDwellSteps,
                                feasibilityWindowPenalty:
                                    feasibilityWindowPenalty(
                                        windows[environment],
                                        required: configuration
                                            .targetFeasibilityDwellSteps),
                                stablePathViolationSteps:
                                    violations[environment],
                                trajectorySteps: horizon,
                                requireStableCarryPath:
                                    configuration.requireStableCarryPath)
                        }
                        let terminalViabilityPenalties = replicaRange.map {
                            environment in
                            let terminalClearancePenalty = configuration
                                .minimumTargetTerminalClearanceMeters.map {
                                    hardMinimumPenalty(
                                        value: clearance[environment],
                                        minimum: $0, scale: 0.01)
                                } ?? 0
                            let downwardVelocityPenalty = configuration
                                .maximumTargetTerminalDownwardBoxVelocityMPS
                                .map {
                                    hardMinimumPenalty(
                                        value: terminalStates[environment]
                                            .manipulation.object
                                            .linearVelocity.z,
                                        minimum: -$0, scale: 0.05)
                                } ?? 0
                            return max(
                                terminalClearancePenalty,
                                downwardVelocityPenalty)
                        }
                        return TargetDiscoveryCandidate(
                            parameters: parameters[candidate],
                            loss: max(
                                evaluations.map(\.bottleneckLoss).max()!,
                                terminalViabilityPenalties.max()!,
                                replicaRange.map {
                                    targetEndpointFootUnloadingPenalties[$0]
                                }.max()!,
                                replicaRange.map {
                                    firstControlTerminalPenalties[$0]
                                }.max()!),
                            maximumNormalizedError: max(
                                evaluations.map(
                                    \.maximumNormalizedError).max()!,
                                terminalViabilityPenalties.max()!,
                                replicaRange.map {
                                    targetEndpointFootUnloadingPenalties[$0]
                                }.max()!,
                                replicaRange.map {
                                    firstControlTerminalPenalties[$0]
                                }.max()!),
                            bestStep: horizon,
                            trajectoryStart: trajectoryStart,
                            firstControlSafe: replicaRange.allSatisfy {
                                firstControlSafe[$0]
                            },
                            commitPathSafe: replicaRange.allSatisfy {
                                firstViolationSteps[$0].map {
                                    $0 > requiredSafePrefixSteps
                                } ?? true
                            },
                            predictedPathSafe: replicaRange.allSatisfy {
                                violations[$0] == 0
                            },
                            terminalGoalFeasible:
                                terminalGoalComponents.values.allSatisfy {
                                    $0
                                },
                            terminalGoalComponents: terminalGoalComponents,
                            terminalRecoveryViable:
                                terminalViabilityPenalties.allSatisfy {
                                    $0 == 0
                                },
                            phaseStep: phaseStep,
                            maximumClearanceMeters: replicaRange.map {
                                maximumClearances[$0]
                            }.min()!,
                            maximumCarryDistanceMeters: replicaRange.map {
                                maximumCarryDistances[$0]
                            }.min()!,
                            maximumStableCarryDistanceMeters: replicaRange.map {
                                maximumStableCarryDistances[$0]
                            }.min()!,
                            maximumDestinationProgressMeters:
                                replicaRange.map {
                                    maximumDestinationProgresses[$0]
                                }.min()!,
                            maximumRootDestinationProgressMeters:
                                replicaRange.map {
                                    maximumRootDestinationProgresses[$0]
                                }.min()!,
                            maximumLoadedTouchdowns:
                                replicaRange.map {
                                    maximumLoadedTouchdownCounts[$0]
                                }.min()!,
                            maximumLoadedAlternatingSteps:
                                replicaRange.map {
                                    maximumLoadedAlternatingStepCounts[$0]
                                }.min()!,
                            maximumSwingFootLiftMeters:
                                replicaRange.map {
                                    maximumSwingFootLifts[$0]
                                }.min()!,
                            maximumLoadedFootAirTimeSeconds:
                                replicaRange.map {
                                    maximumLoadedFootAirTimes[$0]
                                }.min()!,
                            maximumFootUnloadingFraction:
                                replicaRange.map {
                                    maximumFootUnloadingFractions[$0]
                                }.min()!,
                            maximumGraspQuality: replicaRange.map {
                                maximumGraspQualities[$0]
                            }.min()!,
                            maximumFeasibilityDwellSteps: replicaRange.map {
                                maximumDwells[$0]
                            }.min()!,
                            maximumPredicateDwellSteps:
                                robustMaximumPredicateDwells,
                            firstStablePathViolationStep:
                                replicaRange.compactMap {
                                    firstViolationSteps[$0]
                                }.min(),
                            finalCarryDistanceMeters: carry[firstReplica],
                            finalPlacementDistanceMeters:
                                placement[firstReplica],
                            finalDestinationProgressMeters:
                                initialPlacementDistances[firstReplica]
                                    - placement[firstReplica],
                            finalRootDestinationProgressMeters:
                                initialRootDestinationDistances[firstReplica]
                                    - runtime.result.metrics[
                                        "state/root_destination_distance_m"]![
                                            firstReplica],
                            finalLoadedTouchdowns: max(
                                0, Int(runtime.result.metrics[
                                    "state/loaded_touchdowns"]![firstReplica]
                                    - initialTouchdowns[firstReplica])),
                            finalLoadedAlternatingSteps: max(
                                0, Int(runtime.result.metrics[
                                    "state/loaded_alternating_steps"]![
                                        firstReplica]
                                    - initialAlternatingSteps[firstReplica])),
                            finalClearanceMeters: clearance[firstReplica],
                            finalGraspQuality:
                                terminalFlags[firstReplica].graspQuality,
                            finalBoxVerticalVelocityMPS:
                                terminalStates[firstReplica].manipulation
                                    .object.linearVelocity.z)
                    }
                }

                func robustlyAggregated(
                    _ replicas: [TargetDiscoveryCandidate],
                    parameters: [Float], phaseStep: Int = 0
                ) -> TargetDiscoveryCandidate {
                    precondition(!replicas.isEmpty)
                    let requiredFraction = configuration
                        .recedingValidationMinimumSuccessFraction
                    let requiresFullHorizonSafety =
                        configuration.recedingSafetyLookaheadSteps == nil
                    let passes: (TargetDiscoveryCandidate) -> Bool = {
                        $0.firstControlSafe
                            && $0.commitPathSafe
                            && $0.terminalGoalFeasible
                            && (!requiresFullHorizonSafety
                                || $0.terminalRecoveryViable)
                    }
                    let validation = robustValidationSummary(
                        replicaPasses: replicas.map(passes),
                        minimumSuccessFraction: requiredFraction)
                    let lowerIndex = validation.lowerQuantileIndex
                    let upperIndex = validation.upperQuantileIndex
                    func lowerQuantile(_ values: [Float]) -> Float {
                        values.sorted()[lowerIndex]
                    }
                    func upperQuantile(_ values: [Float]) -> Float {
                        values.sorted()[upperIndex]
                    }
                    func lowerQuantile(_ values: [Int]) -> Int {
                        values.sorted()[lowerIndex]
                    }
                    func robustlyPassed(
                        _ predicate: (TargetDiscoveryCandidate) -> Bool
                    ) -> Bool {
                        robustValidationSummary(
                            replicaPasses: replicas.map(predicate),
                            minimumSuccessFraction: requiredFraction).passed
                    }
                    // Row zero is the branch subsequently committed and
                    // shown in Policy Replay. Replica agreement may reject
                    // that branch, but it can never make a failing row-zero
                    // trajectory eligible for commitment.
                    let exact = replicas[0]
                    let predicateDwells = (0..<9).map { predicate in
                        lowerQuantile(replicas.map {
                            $0.maximumPredicateDwellSteps![predicate]
                        })
                    }
                    return TargetDiscoveryCandidate(
                        parameters: parameters,
                        loss: upperQuantile(replicas.map(\.loss)),
                        maximumNormalizedError: upperQuantile(replicas.map(
                            \.maximumNormalizedError)),
                        bestStep: exact.bestStep,
                        trajectoryStart: exact.trajectoryStart,
                        firstControlSafe:
                            robustlyPassed(\.firstControlSafe),
                        commitPathSafe:
                            robustlyPassed(\.commitPathSafe),
                        predictedPathSafe:
                            robustlyPassed(\.predictedPathSafe),
                        terminalGoalFeasible:
                            robustlyPassed(\.terminalGoalFeasible),
                        terminalRecoveryViable:
                            robustlyPassed(\.terminalRecoveryViable),
                        jointValidationPassed: validation.passed,
                        validationSuccessFraction:
                            validation.successFraction,
                        phaseStep: phaseStep,
                        maximumClearanceMeters: lowerQuantile(replicas.map {
                            $0.maximumClearanceMeters!
                        }),
                        maximumCarryDistanceMeters: lowerQuantile(replicas.map {
                            $0.maximumCarryDistanceMeters!
                        }),
                        maximumStableCarryDistanceMeters: lowerQuantile(
                            replicas.map {
                            $0.maximumStableCarryDistanceMeters!
                        }),
                        maximumDestinationProgressMeters: lowerQuantile(
                            replicas.map {
                            $0.maximumDestinationProgressMeters!
                        }),
                        maximumRootDestinationProgressMeters: lowerQuantile(
                            replicas.map {
                            $0.maximumRootDestinationProgressMeters!
                        }),
                        maximumLoadedTouchdowns: lowerQuantile(replicas.map {
                            $0.maximumLoadedTouchdowns!
                        }),
                        maximumLoadedAlternatingSteps: lowerQuantile(
                            replicas.map {
                            $0.maximumLoadedAlternatingSteps!
                        }),
                        maximumSwingFootLiftMeters: lowerQuantile(replicas.map {
                            $0.maximumSwingFootLiftMeters!
                        }),
                        maximumLoadedFootAirTimeSeconds: lowerQuantile(
                            replicas.map {
                            $0.maximumLoadedFootAirTimeSeconds!
                        }),
                        maximumFootUnloadingFraction: lowerQuantile(
                            replicas.map {
                            $0.maximumFootUnloadingFraction!
                        }),
                        maximumGraspQuality: lowerQuantile(replicas.map {
                            $0.maximumGraspQuality!
                        }),
                        maximumFeasibilityDwellSteps: lowerQuantile(
                            replicas.map {
                            $0.maximumFeasibilityDwellSteps!
                        }),
                        maximumPredicateDwellSteps: predicateDwells,
                        firstStablePathViolationStep:
                            exact.firstStablePathViolationStep,
                        finalCarryDistanceMeters:
                            exact.finalCarryDistanceMeters,
                        finalPlacementDistanceMeters:
                            exact.finalPlacementDistanceMeters,
                        finalDestinationProgressMeters:
                            exact.finalDestinationProgressMeters,
                        finalRootDestinationProgressMeters:
                            exact.finalRootDestinationProgressMeters,
                        finalLoadedTouchdowns:
                            exact.finalLoadedTouchdowns,
                        finalLoadedAlternatingSteps:
                            exact.finalLoadedAlternatingSteps,
                        finalClearanceMeters:
                            exact.finalClearanceMeters,
                        finalGraspQuality:
                            exact.finalGraspQuality,
                        finalBoxVerticalVelocityMPS:
                            exact.finalBoxVerticalVelocityMPS)
                }

                let generationCount =
                    archivedTargetTrajectorySequence != nil
                        || reusingCertifiedPlan
                    ? 0 : configuration.targetDiscoveryGenerations
                for generation in 0..<generationCount {
                    let searchBlock: TargetDiscoverySearchBlock
                    if configuration.targetDiscoveryBlockCoordinateSearch {
                        let phase = min(
                            2,
                            3 * generation
                                / configuration
                                    .targetDiscoveryGenerations)
                        searchBlock = switch phase {
                            case 0: .lowerBody
                            case 1: .upperBody
                            default: .joint
                        }
                    } else {
                        searchBlock = .configured
                    }
                    let transform = cholesky(covariance)
                    var parameters = [[Float]]()
                    parameters.reserveCapacity(count)
                    let locomotionMean = locomotionProposal(from: mean)
                    let samplesLocomotionProposal =
                        configuration.recedingLocomotionBlendProposal != nil
                            && searchBlock != .upperBody
                    let swingFrontierMean =
                        bestSwingFrontierCandidate.map {
                            structuredTargetDiscoveryTrajectory(
                                projectedTargetDiscoveryTrajectory(
                                    $0.parameters,
                                    onto: mean,
                                block: searchBlock))
                        }
                    let stableSwingMean =
                        bestStableSwingCandidate.map {
                            structuredTargetDiscoveryTrajectory(
                                projectedTargetDiscoveryTrajectory(
                                    $0.parameters,
                                    onto: mean,
                                block: searchBlock))
                        }
                    let feasibilityFrontierMean =
                        bestFeasibilityFrontierCandidate.flatMap {
                            ($0.maximumFeasibilityDwellSteps ?? 0) > 0
                                ? structuredTargetDiscoveryTrajectory(
                                    projectedTargetDiscoveryTrajectory(
                                        $0.parameters,
                                        onto: mean,
                                        block: searchBlock))
                                : nil
                        }
                    for index in 0..<count {
                        if index == 0 {
                            parameters.append(mean)
                        } else if index == 1, let overallBest {
                            parameters.append(
                                structuredTargetDiscoveryTrajectory(
                                    projectedTargetDiscoveryTrajectory(
                                        overallBest.parameters,
                                        onto: mean,
                                        block: searchBlock)))
                        } else if index == 2,
                                  samplesLocomotionProposal {
                            parameters.append(locomotionMean)
                        } else if index == 3,
                                  let swingFrontierMean {
                            parameters.append(swingFrontierMean)
                        } else if index == 4,
                                  let stableSwingMean {
                            parameters.append(stableSwingMean)
                        } else if index == 5,
                                  let feasibilityFrontierMean {
                            parameters.append(feasibilityFrontierMean)
                        } else {
                            let repairsFeasibilityFrontier =
                                feasibilityFrontierMean != nil
                                    && index % 6 == 1
                            let repairsStableSwing =
                                !repairsFeasibilityFrontier
                                    && stableSwingMean != nil
                                    && index % 6 == 3
                            let repairsSwingFrontier =
                                !repairsFeasibilityFrontier
                                    && !repairsStableSwing
                                    && swingFrontierMean != nil
                                    && index % 6 == 5
                            let sampleCenter: [Float]
                            if repairsFeasibilityFrontier,
                               let feasibilityFrontierMean {
                                sampleCenter = feasibilityFrontierMean
                            } else if repairsStableSwing,
                               let stableSwingMean {
                                sampleCenter = stableSwingMean
                            } else if repairsSwingFrontier,
                               let swingFrontierMean {
                                sampleCenter = swingFrontierMean
                            } else {
                                sampleCenter = samplesLocomotionProposal
                                        && index.isMultiple(of: 2)
                                    ? locomotionMean : mean
                            }
                            if repairsFeasibilityFrontier
                                || repairsStableSwing
                                || repairsSwingFrontier {
                                parameters.append(sampleIndependent(
                                    around: sampleCenter,
                                    standardDeviation: configuration
                                        .targetDiscoveryInitialStandardDeviation,
                                    block: .upperBody))
                            } else {
                                parameters.append(sample(
                                    around: sampleCenter,
                                    transform: transform,
                                    block: searchBlock))
                            }
                        }
                    }
                    // Infeasible candidates remain useful for moving CEM
                    // toward the constraint boundary, but only a separately
                    // tracked feasible incumbent may ever be committed. This
                    // avoids both failure modes: aborting before CEM can cross
                    // a poor initial distribution, and applying a "least bad"
                    // action that walks the real state into a collision.
                    let candidates = try evaluateReceding(
                        parameters,
                        phaseStep: searchPhaseStep,
                        trajectoryStart: recedingTrajectoryStart)
                        .sorted { pathCandidateIsBetter($0, than: $1) }
                    for candidate in candidates {
                        speculativeMaximumRootDestinationProgress = max(
                            speculativeMaximumRootDestinationProgress,
                            candidate.maximumRootDestinationProgressMeters
                                ?? -.infinity)
                        speculativeMaximumLoadedTouchdowns = max(
                            speculativeMaximumLoadedTouchdowns,
                            candidate.maximumLoadedTouchdowns ?? 0)
                        speculativeMaximumLoadedAlternatingSteps = max(
                            speculativeMaximumLoadedAlternatingSteps,
                            candidate.maximumLoadedAlternatingSteps ?? 0)
                        speculativeMaximumSwingFootLift = max(
                            speculativeMaximumSwingFootLift,
                            candidate.maximumSwingFootLiftMeters ?? 0)
                        speculativeMaximumLoadedFootAirTime = max(
                            speculativeMaximumLoadedFootAirTime,
                            candidate.maximumLoadedFootAirTimeSeconds ?? 0)
                        speculativeMaximumFootUnloadingFraction = max(
                            speculativeMaximumFootUnloadingFraction,
                            candidate.maximumFootUnloadingFraction ?? 0)
                        speculativeMaximumFeasibilityDwell = max(
                            speculativeMaximumFeasibilityDwell,
                            candidate.maximumFeasibilityDwellSteps ?? 0)
                        if let candidateDwells =
                                candidate.maximumPredicateDwellSteps {
                            for predicate in 0..<9 {
                                speculativeMaximumPredicateDwells[predicate] =
                                    max(
                                        speculativeMaximumPredicateDwells[
                                            predicate],
                                        candidateDwells[predicate])
                            }
                        }
                        if isBetterStableSwing(
                            candidate,
                            than: bestStableSwingCandidate) {
                            bestStableSwingCandidate = candidate
                        }
                        if swingFrontierIsBetter(
                            swingFrontierScore(candidate),
                            than: bestSwingFrontierCandidate.map(
                                swingFrontierScore)) {
                            bestSwingFrontierCandidate = candidate
                        }
                        if feasibilityFrontierIsBetter(
                            feasibilityFrontierScore(candidate),
                            than: bestFeasibilityFrontierCandidate.map(
                                feasibilityFrontierScore)) {
                            bestFeasibilityFrontierCandidate = candidate
                        }
                    }
                    guard let generationBest = candidates.first,
                          generationBest.loss.isFinite else {
                        throw RLEnvironmentError.invalidConfiguration(
                            "non-finite humanoid-box receding-horizon search")
                    }
                    let validationParameters =
                        Self.prioritizedValidationParameters(
                            lossRanked: candidates.map(\.parameters),
                            frontiers: [
                                bestFeasibilityFrontierCandidate.flatMap {
                                    ($0.maximumFeasibilityDwellSteps ?? 0) > 0
                                        ? $0.parameters : nil
                                },
                                bestStableSwingCandidate?.parameters,
                                bestSwingFrontierCandidate?.parameters,
                            ].compactMap { $0 },
                            limit: configuration
                                .recedingValidationCandidateCount)
                    let validated = try validationParameters.map {
                        parameters in
                        let repeated = [[Float]](
                            repeating: parameters, count: count)
                        let replicas = try evaluateReceding(
                            repeated,
                            phaseStep: searchPhaseStep,
                            trajectoryStart: recedingTrajectoryStart)
                        return robustlyAggregated(
                            Array(replicas.prefix(validationReplicaCount)),
                            parameters: parameters,
                            phaseStep: searchPhaseStep)
                    }.sorted { pathCandidateIsBetter($0, than: $1) }
                    guard let validatedBest = validated.first,
                          validatedBest.loss.isFinite else {
                        throw RLEnvironmentError.invalidConfiguration(
                            "non-finite humanoid-box robust validation")
                    }
                    if overallNearMiss == nil
                        || pathCandidateIsBetter(
                            validatedBest, than: overallNearMiss!) {
                        overallNearMiss = validatedBest
                    }
                    if let safeBest = validated.first(where: {
                        $0.jointValidationPassed
                            && $0.firstControlSafe
                            && $0.commitPathSafe
                            && $0.terminalGoalFeasible
                            && (configuration
                                    .recedingSafetyLookaheadSteps != nil
                                || $0.terminalRecoveryViable)
                    }), overallBest == nil
                        || pathCandidateIsBetter(
                            safeBest, than: overallBest!) {
                        overallBest = safeBest
                    }
                    let eliteCount = min(candidates.count, max(
                        2, Int(Float(count) * configuration.eliteFraction)))
                    let elites = candidates.prefix(eliteCount)
                    var nextMean = [Float](repeating: 0,
                                           count: parameterCount)
                    for elite in elites {
                        for index in 0..<parameterCount {
                            nextMean[index] += elite.parameters[index]
                        }
                    }
                    for index in 0..<parameterCount {
                        nextMean[index] /= Float(eliteCount)
                    }
                    var nextCovariance = [[Float]](
                        repeating: [Float](repeating: 0,
                                           count: parameterCount),
                        count: parameterCount)
                    for elite in elites {
                        for row in 0..<parameterCount {
                            for column in 0..<parameterCount {
                                nextCovariance[row][column] +=
                                    (elite.parameters[row] - nextMean[row])
                                    * (elite.parameters[column]
                                        - nextMean[column])
                            }
                        }
                    }
                    for row in 0..<parameterCount {
                        for column in 0..<parameterCount {
                            nextCovariance[row][column] /= Float(eliteCount)
                        }
                        nextCovariance[row][row] = max(
                            nextCovariance[row][row], 0.02 * 0.02)
                    }
                    mean = zip(mean, nextMean).map {
                        0.25 * $0.0 + 0.75 * $0.1
                    }
                    covariance = nextCovariance
                }
                if overallBest == nil,
                   let fallbackPlan,
                   fallbackPlan.phaseStep < horizon {
                    let repeated = [[Float]](
                        repeating: fallbackPlan.parameters, count: count)
                    let replicas = try evaluateReceding(
                        repeated,
                        phaseStep: fallbackPlan.phaseStep,
                        trajectoryStart: fallbackPlan.trajectoryStart)
                    let fallback = robustlyAggregated(
                        Array(replicas.prefix(validationReplicaCount)),
                        parameters: fallbackPlan.parameters,
                        phaseStep: fallbackPlan.phaseStep)
                    if fallback.jointValidationPassed
                        && fallback.firstControlSafe
                        && fallback.commitPathSafe
                        && fallback.terminalGoalFeasible
                        && (configuration
                                .recedingSafetyLookaheadSteps != nil
                            || fallback.terminalRecoveryViable) {
                        overallBest = fallback
                    } else if overallNearMiss == nil
                                || pathCandidateIsBetter(
                                    fallback, than: overallNearMiss!) {
                        overallNearMiss = fallback
                    }
                }
                guard let selected = overallBest else {
                    let near = overallNearMiss
                    func predictedTrace(
                        parameters: [Float], phaseStep: Int,
                        trajectoryStart: StructuredTrajectoryBoundary?
                    ) throws -> [HumanoidBoxPhysicalFlowTraceSample] {
                        restoreCommittedState()
                        let repeated = [[Float]](
                            repeating: parameters, count: runtimeCount)
                        var trace = [HumanoidBoxPhysicalFlowTraceSample]()
                        let activePlanSteps = horizon - phaseStep
                        let evaluationSteps = recedingEvaluationSteps(
                            horizon: horizon,
                            phaseStep: phaseStep,
                            terminalHoldSteps:
                                configuration.recedingTerminalHoldSteps)
                        trace.reserveCapacity(evaluationSteps + 1)
                        var maximumLift = maximumSwingFootLift
                        var maximumAir = maximumLoadedFootAirTime
                        var maximumUnloading =
                            maximumFootUnloadingFraction
                        trace.append(currentTraceSample(
                            step: committedStep))
                        for predictedStep in 0..<evaluationSteps {
                            let actions = predictedStep < activePlanSteps
                                ? try applyTrajectory(
                                    repeated,
                                    step: phaseStep + predictedStep,
                                    denominator: horizon,
                                    task: runtime.task,
                                    observation: runtime.observation,
                                    forwardOnlyBaseCommand: configuration
                                        .recedingForwardOnlyBaseCommand,
                                    holonomicBaseCommand: configuration
                                        .recedingHolonomicBaseCommand,
                                    locomotionCheckpointDirectory:
                                        configuration
                                            .recedingLocomotionCheckpointDirectory,
                                    locomotionCommandSpeed: configuration
                                        .recedingLocomotionCommandSpeed,
                                    trajectoryStart: trajectoryStart,
                                    graspFeedback: graspFeedback)
                                : try applyTrajectory(
                                    repeated, step: horizon - 1,
                                    denominator: horizon,
                                    task: runtime.task,
                                    observation: runtime.observation,
                                    applyLegAndTorsoTrajectory: false,
                                    trajectoryStart: trajectoryStart,
                                    graspFeedback: graspFeedback)
                            try runtime.task.step(
                                actions: actions, into: &runtime.result)
                            runtime.observation =
                                runtime.result.observations
                            maximumLift = max(
                                maximumLift, runtime.result.metrics[
                                    "state/maximum_loaded_swing_clearance_m"]![
                                        0])
                            maximumAir = max(
                                maximumAir, runtime.result.metrics[
                                    "state/maximum_loaded_foot_air_time_s"]![
                                        0])
                            maximumUnloading = max(
                                maximumUnloading, runtime.result.metrics[
                                    "state/foot_unloading_fraction"]![0])
                            var sample = currentTraceSample(
                                step: committedStep + predictedStep + 1,
                                appliedActions: actions)
                            sample.maximumSwingFootLiftMeters =
                                maximumLift
                            sample.maximumLoadedFootAirTimeSeconds =
                                maximumAir
                            sample.maximumFootUnloadingFraction =
                                maximumUnloading
                            trace.append(sample)
                        }
                        return trace
                    }
                    let nearMissPredictedTrace = try near.map {
                        try predictedTrace(
                            parameters: $0.parameters,
                            phaseStep: $0.phaseStep,
                            trajectoryStart: $0.trajectoryStart)
                    }
                    let bestStableSwingTrace =
                        try bestStableSwingCandidate.map {
                            try predictedTrace(
                                parameters: $0.parameters,
                                phaseStep: $0.phaseStep,
                                trajectoryStart: $0.trajectoryStart)
                        }
                    let bestSwingFrontierTrace =
                        try bestSwingFrontierCandidate.map {
                            try predictedTrace(
                                parameters: $0.parameters,
                                phaseStep: $0.phaseStep,
                                trajectoryStart: $0.trajectoryStart)
                        }
                    let bestFeasibilityFrontierTrace =
                        try bestFeasibilityFrontierCandidate.map {
                            try predictedTrace(
                                parameters: $0.parameters,
                                phaseStep: $0.phaseStep,
                                trajectoryStart: $0.trajectoryStart)
                        }
                    let nearPredicates = near?
                        .maximumPredicateDwellSteps
                        ?? [Int](repeating: 0, count: 9)
                    let combinedPredicateDwells = zip(
                        zip(maximumPredicateDwells, nearPredicates).map(max),
                        speculativeMaximumPredicateDwells).map(max)
                    func finite(
                        _ value: Float?, fallback: Float
                    ) -> Float {
                        guard let value, value.isFinite else {
                            return fallback
                        }
                        return value
                    }
                    let source = committedTrace.last!
                    throw HumanoidBoxPhysicalFlowTargetFailure(
                        experiment:
                            "humanoid-box-receding-search-failure-v0",
                        seed: configuration.seed,
                        optimizerSeed: configuration.optimizerSeed,
                        targetGeneratingTrajectory: seedTrajectory,
                        targetGeneratingTrajectorySequence: selectedSequence,
                        targetGeneratingTrajectorySequencePhaseSteps:
                            selectedSequencePhaseSteps,
                        derivedStageContinuesFromSourceTerminal:
                            configuration.continueTrajectoryFromSourceTerminal
                                ? true : nil,
                        committedTrace: committedTrace,
                        targetDiscoveryPopulationSize: count,
                        targetDiscoveryGenerations: configuration
                            .targetDiscoveryGenerations,
                        targetDiscoveryCandidateRollouts:
                            targetDiscoveryCandidateRollouts,
                        targetDiscoveryTiedArmKnots:
                            configuration.targetDiscoveryTiedArmKnots
                                ? true : nil,
                        targetDiscoveryBlockCoordinateSearch:
                            configuration
                                .targetDiscoveryBlockCoordinateSearch
                                    ? true : nil,
                        targetDiscoverySeededFromSourceTerminalUpperBody:
                            seedsFromSourceTerminalTrajectory ? true : nil,
                        targetDiscoveryHeldSourceTerminalUpperBody:
                            holdsSourceTerminalUpperBody ? true : nil,
                        targetDiscoveryPreservedProvidedUpperBodySeed:
                            preservesProvidedUpperBodySeed ? true : nil,
                        recedingValidationMinimumSuccessFraction:
                            configuration
                                .recedingValidationMinimumSuccessFraction,
                        nearMissLoss: near?.loss,
                        nearMissValidationSuccessFraction:
                            near?.validationSuccessFraction,
                        nearMissFirstControlSafe:
                            near?.firstControlSafe,
                        nearMissCommitPathSafe:
                            near?.commitPathSafe,
                        nearMissPredictedPathSafe:
                            near?.predictedPathSafe,
                        nearMissTerminalGoalFeasible:
                            near?.terminalGoalFeasible,
                        nearMissTerminalRecoveryViable:
                            near?.terminalRecoveryViable,
                        nearMissParameters: near?.parameters,
                        nearMissPhaseStep: near?.phaseStep,
                        nearMissTrajectoryStart:
                            near?.trajectoryStart,
                        nearMissPredictedTrace:
                            nearMissPredictedTrace,
                        bestStableSwingLoss:
                            bestStableSwingCandidate?.loss,
                        bestStableSwingTerminalGoalFeasible:
                            bestStableSwingCandidate?
                                .terminalGoalFeasible,
                        bestStableSwingTerminalRecoveryViable:
                            bestStableSwingCandidate?
                                .terminalRecoveryViable,
                        bestStableSwingParameters:
                            bestStableSwingCandidate?.parameters,
                        bestStableSwingTrace:
                            bestStableSwingTrace,
                        bestSwingFrontierLoss:
                            bestSwingFrontierCandidate?.loss,
                        bestSwingFrontierFirstControlSafe:
                            bestSwingFrontierCandidate?
                                .firstControlSafe,
                        bestSwingFrontierPredictedPathSafe:
                            bestSwingFrontierCandidate?
                                .predictedPathSafe,
                        bestSwingFrontierTerminalGoalFeasible:
                            bestSwingFrontierCandidate?
                                .terminalGoalFeasible,
                        bestSwingFrontierTerminalRecoveryViable:
                            bestSwingFrontierCandidate?
                                .terminalRecoveryViable,
                        bestSwingFrontierParameters:
                            bestSwingFrontierCandidate?.parameters,
                        bestSwingFrontierTrace:
                            bestSwingFrontierTrace,
                        bestFeasibilityFrontierLoss:
                            bestFeasibilityFrontierCandidate?.loss,
                        bestFeasibilityFrontierDwellSteps:
                            bestFeasibilityFrontierCandidate?
                                .maximumFeasibilityDwellSteps,
                        bestFeasibilityFrontierTerminalGoalFeasible:
                            bestFeasibilityFrontierCandidate?
                                .terminalGoalFeasible,
                        bestFeasibilityFrontierTerminalGoalComponents:
                            bestFeasibilityFrontierCandidate?
                                .terminalGoalComponents,
                        bestFeasibilityFrontierTerminalRecoveryViable:
                            bestFeasibilityFrontierCandidate?
                                .terminalRecoveryViable,
                        bestFeasibilityFrontierParameters:
                            bestFeasibilityFrontierCandidate?.parameters,
                        bestFeasibilityFrontierTrace:
                            bestFeasibilityFrontierTrace,
                        recedingValidationReplicaCount:
                            validationReplicaCount,
                        recedingLocomotionCheckpointDirectory:
                            configuration
                                .recedingLocomotionCheckpointDirectory,
                        recedingLocomotionCommandSpeed:
                            configuration.recedingLocomotionCommandSpeed,
                        recedingLocomotionBlendProposal:
                            configuration
                                .recedingLocomotionBlendProposal,
                        recedingLocomotionZeroResidualProposal:
                            configuration
                                .recedingLocomotionZeroResidualProposal
                                    ? true : nil,
                        recedingForwardOnlyBaseCommand:
                            configuration.recedingForwardOnlyBaseCommand,
                        recedingHolonomicBaseCommand:
                            configuration.recedingHolonomicBaseCommand,
                        targetGenerationSteps: horizon,
                        recedingInitialPhaseStep:
                            configuration.recedingInitialPhaseStep,
                        recedingControlHorizonSteps:
                            configuration.recedingControlHorizonSteps,
                        recedingSafetyLookaheadSteps:
                            configuration.recedingSafetyLookaheadSteps,
                        recedingTerminalHoldSteps: configuration
                            .recedingTerminalHoldSteps,
                        targetExecutionSteps: targetExecutionSteps,
                        sourceStages: sourceStages,
                        sourceWarmupAppliedActions:
                            sourceWarmupAppliedActions,
                        sourceAppliedActions: sourceAppliedActions,
                        legBlendKnotCount:
                            configuration.legBlendKnotCount,
                        legResidualKnotCount:
                            configuration.legResidualKnotCount,
                        maximumLegResidualAction:
                            configuration.maximumLegResidualAction,
                        torsoResidualKnotCount:
                            configuration.torsoResidualKnotCount,
                        maximumTorsoResidualAction:
                            configuration.maximumTorsoResidualAction,
                        armAsymmetryKnotCount:
                            configuration.armAsymmetryKnotCount,
                        maximumArmAsymmetryAction:
                            configuration.maximumArmAsymmetryAction,
                        graspAnchorFeedbackBlend:
                            graspFeedback?.blend,
                        graspAnchorFeedbackVelocityHorizonSeconds:
                            graspFeedback?.velocityHorizonSeconds,
                        graspAnchorFeedbackMaximumActionCorrection:
                            graspFeedback?.maximumActionCorrection,
                        graspAnchorFeedbackInwardPreloadMeters:
                            graspFeedback?.inwardPreloadMeters,
                        leftGraspAnchorBoxLocalMeters:
                            graspFeedback.map {
                                [$0.leftAnchorBoxLocal.x,
                                 $0.leftAnchorBoxLocal.y,
                                 $0.leftAnchorBoxLocal.z]
                            },
                        rightGraspAnchorBoxLocalMeters:
                            graspFeedback.map {
                                [$0.rightAnchorBoxLocal.x,
                                 $0.rightAnchorBoxLocal.y,
                                 $0.rightAnchorBoxLocal.z]
                            },
                        graspAnchorBoxHeightMeters:
                            graspFeedback?.boxHeightMeters,
                        maximumClearanceMeters: max(
                            finite(near?.maximumClearanceMeters,
                                   fallback: source.boxClearanceMeters),
                            finite(maximumClearance,
                                   fallback: source.boxClearanceMeters),
                            source.boxClearanceMeters),
                        maximumCarryDistanceMeters: max(
                            finite(near?.maximumCarryDistanceMeters,
                                   fallback: source.carryDistanceMeters),
                            maximumCarry,
                            source.carryDistanceMeters),
                        maximumStableCarryDistanceMeters: max(
                            finite(near?.maximumStableCarryDistanceMeters,
                                   fallback: source.carryDistanceMeters),
                            maximumStableCarry,
                            source.carryDistanceMeters),
                        maximumDestinationProgressMeters: max(
                            finite(near?
                                .maximumDestinationProgressMeters,
                                fallback:
                                    source.destinationProgressMeters),
                            finite(maximumDestinationProgress,
                                   fallback:
                                    source.destinationProgressMeters),
                            source.destinationProgressMeters),
                        maximumRootDestinationProgressMeters: max(
                            finite(near?
                                .maximumRootDestinationProgressMeters,
                                fallback:
                                    source.rootDestinationProgressMeters
                                        ?? 0),
                            finite(maximumRootDestinationProgress,
                                   fallback:
                                    source.rootDestinationProgressMeters
                                        ?? 0),
                            finite(
                                speculativeMaximumRootDestinationProgress,
                                fallback: 0),
                            source.rootDestinationProgressMeters ?? 0),
                        maximumLoadedTouchdowns: max(
                            near?.maximumLoadedTouchdowns ?? 0,
                            maximumLoadedTouchdowns,
                            speculativeMaximumLoadedTouchdowns,
                            source.loadedTouchdowns ?? 0),
                        maximumLoadedAlternatingSteps: max(
                            near?.maximumLoadedAlternatingSteps ?? 0,
                            maximumLoadedAlternatingSteps,
                            speculativeMaximumLoadedAlternatingSteps,
                            source.loadedAlternatingSteps ?? 0),
                        maximumSwingFootLiftMeters: max(
                            finite(near?.maximumSwingFootLiftMeters,
                                   fallback:
                                    source.maximumSwingFootLiftMeters ?? 0),
                            maximumSwingFootLift,
                            speculativeMaximumSwingFootLift,
                            source.maximumSwingFootLiftMeters ?? 0),
                        maximumLoadedFootAirTimeSeconds: max(
                            finite(
                                near?.maximumLoadedFootAirTimeSeconds,
                                fallback: source
                                    .maximumLoadedFootAirTimeSeconds ?? 0),
                            maximumLoadedFootAirTime,
                            speculativeMaximumLoadedFootAirTime,
                            source.maximumLoadedFootAirTimeSeconds ?? 0),
                        maximumFootUnloadingFraction: max(
                            finite(
                                near?.maximumFootUnloadingFraction,
                                fallback: source
                                    .maximumFootUnloadingFraction ?? 0),
                            maximumFootUnloadingFraction,
                            speculativeMaximumFootUnloadingFraction,
                            source.maximumFootUnloadingFraction ?? 0),
                        maximumFeasibilityDwellSteps: max(
                            maximumCommittedDwell,
                            near?.maximumFeasibilityDwellSteps ?? 0,
                            speculativeMaximumFeasibilityDwell),
                        maximumBilateralDwellSteps:
                            combinedPredicateDwells[0],
                        maximumLoadBearingGraspDwellSteps:
                            combinedPredicateDwells[8],
                        maximumUnsupportedDwellSteps:
                            combinedPredicateDwells[1],
                        maximumPhysicallyLiftedDwellSteps:
                            combinedPredicateDwells[2],
                        maximumUprightDwellSteps:
                            combinedPredicateDwells[3],
                        maximumCarryThresholdDwellSteps:
                            combinedPredicateDwells[4],
                        maximumClearanceThresholdDwellSteps:
                            combinedPredicateDwells[6],
                        maximumGraspQualityThresholdDwellSteps:
                            combinedPredicateDwells[7],
                        maximumGraspQuality: max(
                            finite(near?.maximumGraspQuality,
                                   fallback: source.graspQuality),
                            finite(maximumGraspQuality,
                                   fallback: source.graspQuality),
                            source.graspQuality),
                        firstStablePathViolationStep: [
                            firstPathViolation,
                            near?.firstStablePathViolationStep.map {
                                committedStep + $0
                            },
                        ].compactMap { $0 }.min(),
                        finalCarryDistanceMeters:
                            near?.finalCarryDistanceMeters,
                        finalPlacementDistanceMeters:
                            near?.finalPlacementDistanceMeters,
                        finalDestinationProgressMeters:
                            near?.finalDestinationProgressMeters,
                        finalRootDestinationProgressMeters:
                            near?.finalRootDestinationProgressMeters,
                        finalLoadedTouchdowns:
                            near?.finalLoadedTouchdowns,
                        finalLoadedAlternatingSteps:
                            near?.finalLoadedAlternatingSteps,
                        finalClearanceMeters:
                            near?.finalClearanceMeters,
                        finalGraspQuality:
                            near?.finalGraspQuality,
                        finalBoxVerticalVelocityMPS:
                            near?.finalBoxVerticalVelocityMPS,
                        physicalBalanceGatePassed: false,
                        requiredCarryDistanceMeters:
                            configuration
                                .minimumTargetCarryDistanceMeters,
                        requiredPathCarryDistanceMeters:
                            minimumTargetPathCarryDistance,
                        requiredDestinationProgressMeters:
                            configuration
                                .minimumTargetDestinationProgressMeters,
                        requiredRootDestinationProgressMeters:
                            configuration
                                .minimumTargetRootDestinationProgressMeters,
                        requiredTouchdowns:
                            configuration.minimumTargetTouchdowns,
                        requiredAlternatingSteps:
                            configuration.minimumTargetAlternatingSteps,
                        requiredSwingFootLiftMeters:
                            configuration.minimumTargetSwingFootLiftMeters,
                        objectiveSwingFootLiftMeters:
                            configuration
                                .targetDiscoveryObjectiveSwingFootLiftMeters,
                        requiredFootAirTimeSeconds:
                            configuration.minimumTargetFootAirTimeSeconds,
                        objectiveFootAirTimeSeconds:
                            configuration
                                .targetDiscoveryObjectiveFootAirTimeSeconds,
                        requiredFootUnloadingFraction:
                            configuration
                                .minimumTargetFootUnloadingFraction,
                        objectiveFootUnloadingFraction:
                            configuration
                                .targetDiscoveryObjectiveFootUnloadingFraction,
                        requiredTerminalFootUnloadingFraction:
                            configuration
                                .minimumTargetTerminalFootUnloadingFraction,
                        requiredClearanceMeters:
                            configuration.minimumTargetClearanceMeters,
                        maximumPathDownwardBoxVelocityMPS:
                            configuration
                                .maximumTargetPathDownwardBoxVelocityMPS,
                        requiredTerminalClearanceMeters:
                            configuration
                                .minimumTargetTerminalClearanceMeters,
                        maximumTerminalDownwardBoxVelocityMPS:
                            configuration
                                .maximumTargetTerminalDownwardBoxVelocityMPS,
                        requiredGraspQuality:
                            configuration.minimumTargetGraspQuality,
                        requiredFeasibilityDwellSteps:
                            configuration.targetFeasibilityDwellSteps)
                }
                predictedRecoveryPathSafe =
                    predictedRecoveryPathSafe && selected.predictedPathSafe
                if reusingCertifiedPlan {
                    remainingControlHorizonSteps -= 1
                } else {
                    remainingControlHorizonSteps =
                        Self.recedingControlHorizonRemainder(
                            configuredSteps:
                                configuration.recedingControlHorizonSteps,
                            horizon: horizon,
                            selectedPhaseStep: selected.phaseStep)
                }
                selectedSequence.append(selected.parameters)
                selectedSequencePhaseSteps.append(selected.phaseStep)
                let nextFallbackPhase = selected.phaseStep + 1
                fallbackPlan = nextFallbackPhase < horizon
                    ? (
                        selected.parameters,
                        nextFallbackPhase,
                        selected.trajectoryStart
                    ) : nil
                mean = shiftedWarmStart(
                    selected.parameters, controls: nextFallbackPhase)
                restoreCommittedState()
                let repeated = [[Float]](
                    repeating: selected.parameters, count: runtimeCount)
                let action = try applyTrajectory(
                    repeated, step: selected.phaseStep,
                    denominator: horizon,
                    task: runtime.task,
                    observation: runtime.observation,
                    forwardOnlyBaseCommand: configuration
                        .recedingForwardOnlyBaseCommand,
                    holonomicBaseCommand: configuration
                        .recedingHolonomicBaseCommand,
                    locomotionCheckpointDirectory: configuration
                        .recedingLocomotionCheckpointDirectory,
                    locomotionCommandSpeed: configuration
                        .recedingLocomotionCommandSpeed,
                    trajectoryStart: selected.trajectoryStart,
                    graspFeedback: graspFeedback)
                recedingTrajectoryStart =
                    Self.continuedStructuredTrajectoryBoundary(
                        selected.parameters,
                        progress: Float(selected.phaseStep + 1)
                            / Float(horizon),
                        initial: selected.trajectoryStart,
                        armKnotCount: configuration.trajectoryKnotCount,
                        blendKnotCount: configuration.legBlendKnotCount,
                        legResidualKnotCount:
                            configuration.legResidualKnotCount,
                        maximumLegResidualAction:
                            configuration.maximumLegResidualAction,
                        torsoResidualKnotCount:
                            configuration.torsoResidualKnotCount,
                        maximumTorsoResidualAction:
                            configuration.maximumTorsoResidualAction,
                        armAsymmetryKnotCount:
                            configuration.armAsymmetryKnotCount,
                        maximumArmAsymmetryAction:
                            configuration.maximumArmAsymmetryAction)
                try runtime.task.step(
                    actions: action, into: &runtime.result)
                runtime.observation = runtime.result.observations
                let committedStates = states(runtime.task)
                let committedFlags = flags(
                    task: runtime.task, observation: runtime.observation,
                    result: runtime.result)
                maximumGraspQuality = max(
                    maximumGraspQuality, committedFlags[0].graspQuality)
                let carry = runtime.result.metrics[
                    "state/carry_distance_m"]![0]
                let clearance = runtime.result.metrics[
                    "state/box_clearance_m"]![0]
                let placement = runtime.result.metrics[
                    "state/placement_distance_m"]![0]
                let destinationProgress = initialPlacementDistances[0]
                    - placement
                let rootDestinationProgress =
                    initialRootDestinationDistances[0]
                        - runtime.result.metrics[
                            "state/root_destination_distance_m"]![0]
                let loadedAlternatingSteps = max(
                    0, Int(runtime.result.metrics[
                        "state/loaded_alternating_steps"]![0]
                        - initialAlternatingSteps[0]))
                let loadedTouchdowns = max(
                    0, Int(runtime.result.metrics[
                        "state/loaded_touchdowns"]![0]
                        - initialTouchdowns[0]))
                let swingFootLift = runtime.result.metrics[
                    "state/maximum_loaded_swing_clearance_m"]![0]
                let loadedFootAirTime = runtime.result.metrics[
                    "state/maximum_loaded_foot_air_time_s"]![0]
                let footUnloadingFraction = runtime.result.metrics[
                    "state/foot_unloading_fraction"]![0]
                let rootUp = committedStates[0].humanoid.root.rotation
                    .act(F3(0, 0, 1)).z
                let boxUp = committedStates[0].manipulation.object.rotation
                    .act(F3(0, 0, 1)).z
                let stable = committedFlags[0].loadBearingGrasp
                    && committedFlags[0].unsupported
                    && committedFlags[0].physicallyLifted
                    && rootUp > 0.9 && boxUp > 0.9
                    && clearance >= 0.01 && !committedFlags[0].failed
                    && pathDownwardVelocityFeasible(
                        boxVerticalVelocity: committedStates[0].manipulation
                            .object.linearVelocity.z,
                        maximumDownwardVelocity: configuration
                            .maximumTargetPathDownwardBoxVelocityMPS)
                if !stable && firstPathViolation == nil {
                    firstPathViolation = committedStep + 1
                }
                stablePath = stablePath && stable
                let predicatePasses = [
                    committedFlags[0].bilateral,
                    committedFlags[0].unsupported,
                    committedFlags[0].physicallyLifted,
                    rootUp > 0.9 && boxUp > 0.9
                        && !committedFlags[0].failed,
                    carry >= configuration.minimumTargetCarryDistanceMeters,
                    minimumDestinationProgressPassed(
                        destinationProgress,
                        minimum: configuration
                            .minimumTargetDestinationProgressMeters),
                    clearance >= configuration.minimumTargetClearanceMeters,
                    committedFlags[0].graspQuality
                        >= configuration.minimumTargetGraspQuality,
                    committedFlags[0].loadBearingGrasp,
                ]
                for index in predicatePasses.indices {
                    predicateDwells[index] = predicatePasses[index]
                        ? predicateDwells[index] + 1 : 0
                    maximumPredicateDwells[index] = max(
                        maximumPredicateDwells[index],
                        predicateDwells[index])
                }
                let feasible = stable
                    && carry >= configuration.minimumTargetCarryDistanceMeters
                    && minimumDestinationProgressPassed(
                        destinationProgress,
                        minimum: configuration
                            .minimumTargetDestinationProgressMeters)
                    && rootDestinationProgress >= configuration
                        .minimumTargetRootDestinationProgressMeters
                    && loadedTouchdowns >= configuration
                        .minimumTargetTouchdowns
                    && loadedAlternatingSteps >= configuration
                        .minimumTargetAlternatingSteps
                    && max(maximumSwingFootLift, swingFootLift)
                        >= configuration.minimumTargetSwingFootLiftMeters
                    && max(maximumLoadedFootAirTime, loadedFootAirTime)
                        >= configuration.minimumTargetFootAirTimeSeconds
                    && max(
                        maximumFootUnloadingFraction,
                        footUnloadingFraction) >= configuration
                            .minimumTargetFootUnloadingFraction
                    && (configuration
                        .minimumTargetTerminalFootUnloadingFraction.map {
                            footUnloadingFraction >= $0
                        } ?? true)
                    && clearance >= configuration.minimumTargetClearanceMeters
                    && committedFlags[0].graspQuality
                        >= configuration.minimumTargetGraspQuality
                    && (!configuration.requireStableCarryPath || stablePath)
                committedDwell = feasible ? committedDwell + 1 : 0
                maximumCommittedDwell = max(
                    maximumCommittedDwell, committedDwell)
                maximumClearance = max(maximumClearance, clearance)
                maximumCarry = max(maximumCarry, carry)
                maximumDestinationProgress = max(
                    maximumDestinationProgress, destinationProgress)
                maximumRootDestinationProgress = max(
                    maximumRootDestinationProgress,
                    rootDestinationProgress)
                maximumLoadedAlternatingSteps = max(
                    maximumLoadedAlternatingSteps,
                    loadedAlternatingSteps)
                maximumLoadedTouchdowns = max(
                    maximumLoadedTouchdowns, loadedTouchdowns)
                maximumSwingFootLift = max(
                    maximumSwingFootLift, swingFootLift)
                maximumLoadedFootAirTime = max(
                    maximumLoadedFootAirTime, loadedFootAirTime)
                maximumFootUnloadingFraction = max(
                    maximumFootUnloadingFraction,
                    footUnloadingFraction)
                if stable { maximumStableCarry = max(maximumStableCarry, carry) }
                committedTrace.append(currentTraceSample(
                    step: committedStep + 1,
                    appliedActions: action))
            }

            let terminalStates = states(runtime.task)
            let terminalFlags = flags(
                task: runtime.task, observation: runtime.observation,
                result: runtime.result)
            let carry = runtime.result.metrics["state/carry_distance_m"]!
            let clearance = runtime.result.metrics[
                "state/box_clearance_m"]!
            let placement = runtime.result.metrics[
                "state/placement_distance_m"]!
            let destinationProgress = initialPlacementDistances[0]
                - placement[0]
            let rootDestinationProgress =
                initialRootDestinationDistances[0]
                    - runtime.result.metrics[
                        "state/root_destination_distance_m"]![0]
            let loadedAlternatingSteps = max(
                0, Int(runtime.result.metrics[
                    "state/loaded_alternating_steps"]![0]
                    - initialAlternatingSteps[0]))
            let loadedTouchdowns = max(
                0, Int(runtime.result.metrics[
                    "state/loaded_touchdowns"]![0]
                    - initialTouchdowns[0]))
            let rootUp = terminalStates[0].humanoid.root.rotation
                .act(F3(0, 0, 1)).z
            let boxUp = terminalStates[0].manipulation.object.rotation
                .act(F3(0, 0, 1)).z
            let terminalRecoveryViable =
                (configuration.minimumTargetTerminalClearanceMeters.map {
                    clearance[0] >= $0
                } ?? true)
                && (configuration
                    .maximumTargetTerminalDownwardBoxVelocityMPS.map {
                        terminalStates[0].manipulation.object
                            .linearVelocity.z >= -$0
                    } ?? true)
                && (configuration
                    .minimumTargetTerminalFootUnloadingFraction.map {
                        runtime.result.metrics[
                            "state/foot_unloading_fraction"]![0] >= $0
                    } ?? true)
            guard committedDwell
                    >= configuration.targetFeasibilityDwellSteps,
                  rootDestinationProgress >= configuration
                    .minimumTargetRootDestinationProgressMeters,
                  loadedTouchdowns >= configuration
                    .minimumTargetTouchdowns,
                  loadedAlternatingSteps >= configuration
                    .minimumTargetAlternatingSteps,
                  maximumSwingFootLift >= configuration
                    .minimumTargetSwingFootLiftMeters,
                  maximumLoadedFootAirTime >= configuration
                    .minimumTargetFootAirTimeSeconds,
                  maximumFootUnloadingFraction >= configuration
                    .minimumTargetFootUnloadingFraction,
                  (!configuration.requireStableCarryPath || stablePath),
                  terminalRecoveryViable else {
                throw HumanoidBoxPhysicalFlowTargetFailure(
                    experiment: "humanoid-box-receding-target-failure-v0",
                    seed: configuration.seed,
                    optimizerSeed: configuration.optimizerSeed,
                    targetGeneratingTrajectory: seedTrajectory,
                    targetGeneratingTrajectorySequence: selectedSequence,
                    targetGeneratingTrajectorySequencePhaseSteps:
                        selectedSequencePhaseSteps,
                    derivedStageContinuesFromSourceTerminal:
                        configuration.continueTrajectoryFromSourceTerminal
                            ? true : nil,
                    committedTrace: committedTrace,
                    targetDiscoveryTiedArmKnots:
                        configuration.targetDiscoveryTiedArmKnots
                            ? true : nil,
                    targetDiscoveryBlockCoordinateSearch:
                        configuration.targetDiscoveryBlockCoordinateSearch
                            ? true : nil,
                    targetDiscoverySeededFromSourceTerminalUpperBody:
                        seedsFromSourceTerminalTrajectory ? true : nil,
                    targetDiscoveryHeldSourceTerminalUpperBody:
                        holdsSourceTerminalUpperBody ? true : nil,
                    targetDiscoveryPreservedProvidedUpperBodySeed:
                        preservesProvidedUpperBodySeed ? true : nil,
                    recedingValidationReplicaCount:
                        validationReplicaCount,
                    recedingLocomotionCheckpointDirectory: configuration
                        .recedingLocomotionCheckpointDirectory,
                    recedingLocomotionCommandSpeed: configuration
                        .recedingLocomotionCommandSpeed,
                    recedingLocomotionBlendProposal: configuration
                        .recedingLocomotionBlendProposal,
                    recedingLocomotionZeroResidualProposal:
                        configuration
                            .recedingLocomotionZeroResidualProposal
                                ? true : nil,
                    recedingForwardOnlyBaseCommand: configuration
                        .recedingForwardOnlyBaseCommand,
                    recedingHolonomicBaseCommand: configuration
                        .recedingHolonomicBaseCommand,
                    targetGenerationSteps: horizon,
                    recedingInitialPhaseStep:
                        configuration.recedingInitialPhaseStep,
                    recedingControlHorizonSteps:
                        configuration.recedingControlHorizonSteps,
                    recedingSafetyLookaheadSteps:
                        configuration.recedingSafetyLookaheadSteps,
                    recedingTerminalHoldSteps: configuration
                        .recedingTerminalHoldSteps,
                    targetExecutionSteps: targetExecutionSteps,
                    sourceStages: sourceStages,
                    sourceWarmupAppliedActions:
                        sourceWarmupAppliedActions,
                    sourceAppliedActions: sourceAppliedActions,
                    legBlendKnotCount: configuration.legBlendKnotCount,
                    legResidualKnotCount:
                        configuration.legResidualKnotCount,
                    maximumLegResidualAction:
                        configuration.maximumLegResidualAction,
                    torsoResidualKnotCount:
                        configuration.torsoResidualKnotCount,
                    maximumTorsoResidualAction:
                        configuration.maximumTorsoResidualAction,
                    armAsymmetryKnotCount:
                        configuration.armAsymmetryKnotCount,
                    maximumArmAsymmetryAction:
                        configuration.maximumArmAsymmetryAction,
                    graspAnchorFeedbackBlend:
                        graspFeedback?.blend,
                    graspAnchorFeedbackVelocityHorizonSeconds:
                        graspFeedback?.velocityHorizonSeconds,
                    graspAnchorFeedbackMaximumActionCorrection:
                        graspFeedback?.maximumActionCorrection,
                    graspAnchorFeedbackInwardPreloadMeters:
                        graspFeedback?.inwardPreloadMeters,
                    leftGraspAnchorBoxLocalMeters:
                        graspFeedback.map {
                            [$0.leftAnchorBoxLocal.x,
                             $0.leftAnchorBoxLocal.y,
                             $0.leftAnchorBoxLocal.z]
                        },
                    rightGraspAnchorBoxLocalMeters:
                        graspFeedback.map {
                            [$0.rightAnchorBoxLocal.x,
                             $0.rightAnchorBoxLocal.y,
                             $0.rightAnchorBoxLocal.z]
                        },
                    graspAnchorBoxHeightMeters:
                        graspFeedback?.boxHeightMeters,
                    maximumClearanceMeters: maximumClearance,
                    maximumCarryDistanceMeters: maximumCarry,
                    maximumStableCarryDistanceMeters: maximumStableCarry,
                    maximumDestinationProgressMeters:
                        maximumDestinationProgress,
                    maximumRootDestinationProgressMeters:
                        maximumRootDestinationProgress,
                    maximumLoadedTouchdowns:
                        maximumLoadedTouchdowns,
                    maximumLoadedAlternatingSteps:
                        maximumLoadedAlternatingSteps,
                    maximumSwingFootLiftMeters:
                        maximumSwingFootLift,
                    maximumLoadedFootAirTimeSeconds:
                        maximumLoadedFootAirTime,
                    maximumFootUnloadingFraction:
                        maximumFootUnloadingFraction,
                    maximumFeasibilityDwellSteps: maximumCommittedDwell,
                    maximumBilateralDwellSteps:
                        maximumPredicateDwells[0],
                    maximumLoadBearingGraspDwellSteps:
                        maximumPredicateDwells[8],
                    maximumUnsupportedDwellSteps:
                        maximumPredicateDwells[1],
                    maximumPhysicallyLiftedDwellSteps:
                        maximumPredicateDwells[2],
                    maximumUprightDwellSteps:
                        maximumPredicateDwells[3],
                    maximumCarryThresholdDwellSteps:
                        maximumPredicateDwells[4],
                    maximumClearanceThresholdDwellSteps:
                        maximumPredicateDwells[6],
                    maximumGraspQualityThresholdDwellSteps:
                        maximumPredicateDwells[7],
                    maximumGraspQuality: maximumGraspQuality,
                    firstStablePathViolationStep: firstPathViolation,
                    finalCarryDistanceMeters: carry[0],
                    finalPlacementDistanceMeters: placement[0],
                    finalDestinationProgressMeters: destinationProgress,
                    finalRootDestinationProgressMeters:
                        rootDestinationProgress,
                    finalLoadedTouchdowns:
                        loadedTouchdowns,
                    finalLoadedAlternatingSteps:
                        loadedAlternatingSteps,
                    finalClearanceMeters: clearance[0],
                    finalGraspQuality: terminalFlags[0].graspQuality,
                    finalBoxVerticalVelocityMPS: terminalStates[0]
                        .manipulation.object.linearVelocity.z,
                    physicalBalanceGatePassed: terminalRecoveryViable
                        && stablePath
                        && maximumPredicateDwells[0] >= targetExecutionSteps
                        && maximumPredicateDwells[8] >= targetExecutionSteps
                        && maximumPredicateDwells[1] >= targetExecutionSteps
                        && maximumPredicateDwells[2] >= targetExecutionSteps
                        && maximumPredicateDwells[3] >= targetExecutionSteps
                        && (configuration.minimumTargetGraspQuality <= 0
                            || maximumPredicateDwells[7]
                                >= targetExecutionSteps),
                    requiredCarryDistanceMeters:
                        configuration.minimumTargetCarryDistanceMeters,
                    requiredDestinationProgressMeters: configuration
                        .minimumTargetDestinationProgressMeters,
                    requiredRootDestinationProgressMeters: configuration
                        .minimumTargetRootDestinationProgressMeters,
                    requiredTouchdowns:
                        configuration.minimumTargetTouchdowns,
                    requiredAlternatingSteps:
                        configuration.minimumTargetAlternatingSteps,
                    requiredSwingFootLiftMeters:
                        configuration.minimumTargetSwingFootLiftMeters,
                    objectiveSwingFootLiftMeters:
                        configuration
                            .targetDiscoveryObjectiveSwingFootLiftMeters,
                    requiredFootAirTimeSeconds:
                        configuration.minimumTargetFootAirTimeSeconds,
                    objectiveFootAirTimeSeconds:
                        configuration
                            .targetDiscoveryObjectiveFootAirTimeSeconds,
                    requiredFootUnloadingFraction:
                        configuration.minimumTargetFootUnloadingFraction,
                    objectiveFootUnloadingFraction:
                        configuration
                            .targetDiscoveryObjectiveFootUnloadingFraction,
                    requiredTerminalFootUnloadingFraction:
                        configuration
                            .minimumTargetTerminalFootUnloadingFraction,
                    requiredClearanceMeters:
                        configuration.minimumTargetClearanceMeters,
                    maximumPathDownwardBoxVelocityMPS:
                        configuration
                            .maximumTargetPathDownwardBoxVelocityMPS,
                    requiredTerminalClearanceMeters:
                        configuration
                            .minimumTargetTerminalClearanceMeters,
                    maximumTerminalDownwardBoxVelocityMPS:
                        configuration
                            .maximumTargetTerminalDownwardBoxVelocityMPS,
                    requiredGraspQuality:
                        configuration.minimumTargetGraspQuality,
                    requiredFeasibilityDwellSteps:
                        configuration.targetFeasibilityDwellSteps)
            }
            return Target(
                state: terminalStates[0], flags: terminalFlags[0],
                step: targetExecutionSteps, clearance: clearance[0],
                boxUpright: boxUp, robotUpright: rootUp,
                carryDistance: carry[0],
                placementDistance: placement[0],
                destinationProgress: destinationProgress,
                rootDestinationProgress: rootDestinationProgress,
                loadedTouchdowns: loadedTouchdowns,
                loadedAlternatingSteps: loadedAlternatingSteps,
                maximumSwingFootLift: maximumSwingFootLift,
                maximumLoadedFootAirTime:
                    maximumLoadedFootAirTime,
                maximumFootUnloadingFraction:
                    maximumFootUnloadingFraction,
                terminalFootUnloadingFraction: runtime.result.metrics[
                    "state/foot_unloading_fraction"]![0],
                replicas: terminalStates,
                replicaFlags: terminalFlags,
                replicaCarryDistances: Array(carry),
                replicaClearances: Array(clearance),
                replicaGraspQualities:
                    terminalFlags.map(\.graspQuality),
                replicaFootUnloadingFractions: Array(
                    runtime.result.metrics[
                        "state/foot_unloading_fraction"]!),
                warmupSteps: runtime.warmupSteps,
                simulatedPreparationSteps: 0,
                sourceReplaySuccessFraction:
                    runtime.sourceReplaySuccessFraction,
                stableCarryPath: stablePath,
                predictedRecoveryPathSafe: predictedRecoveryPathSafe,
                generatingTrajectorySequence: selectedSequence,
                generatingTrajectorySequencePhaseSteps:
                    selectedSequencePhaseSteps,
                committedTrace: committedTrace,
                graspFeedback: graspFeedback)
        }

        let target = configuration.recedingHorizonSteps > 0
            ? try generateRecedingTarget(using: selectedTargetTrajectory)
            : try generateTarget(using: selectedTargetTrajectory)
        simulatedEnvironmentControlSteps += target.replicas.count
            * (target.simulatedPreparationSteps + targetExecutionSteps)

        func evaluate(
            _ parameters: [[Float]],
            actionNoiseStandardDeviation: Float = 0,
            actionNoiseSeedOffset: UInt64 = 0
        ) throws -> [Candidate] {
            precondition(!parameters.isEmpty)
            precondition(actionNoiseStandardDeviation.isFinite
                && actionNoiseStandardDeviation >= 0)
            var runtime = try prepareSource(count: parameters.count)
            var actionNoiseGenerators = parameters.indices.map {
                ProbeRandomNumberGenerator(
                    seed: optimizerSeed &+ 0xA671_0A5E
                        &+ actionNoiseSeedOffset &* 0xD1B5_4A32
                        &+ UInt64($0) &* 0x9E37_79B9)
            }
            var stableCarryPaths = [Bool](
                repeating: true, count: parameters.count)
            var stablePathViolationSteps = [Int](
                repeating: 0, count: parameters.count)
            var endpointDwells = [Int](
                repeating: 0, count: parameters.count)
            var endpointFeasibilityWindows = [[Float]](
                repeating: [], count: parameters.count)
            for step in 0..<target.step {
                var actions = try applyTrajectory(
                    parameters, step: step,
                    denominator: configuration
                        .candidateTrajectoryDurationSteps ?? target.step,
                    task: runtime.task, observation: runtime.observation,
                    trajectoryStart: currentTrajectoryStart,
                    graspFeedback: target.graspFeedback)
                if actionNoiseStandardDeviation > 0 {
                    for environment in parameters.indices {
                        let base = environment
                            * runtime.task.spec.action.elementCount
                        for action in 0..<runtime.task.spec.action.elementCount {
                            actions.values[base + action] = simd_clamp(
                                actions.values[base + action]
                                    + actionNoiseStandardDeviation
                                    * actionNoiseGenerators[environment]
                                        .normal(),
                                -0.999, 0.999)
                        }
                    }
                }
                try runtime.task.step(actions: actions, into: &runtime.result)
                runtime.observation = runtime.result.observations
                let intermediateStates = states(runtime.task)
                let intermediateFlags = flags(
                    task: runtime.task,
                    observation: runtime.observation,
                    result: runtime.result)
                let intermediateClearances = runtime.result.metrics[
                    "state/box_clearance_m"]!
                let intermediateCarry = runtime.result.metrics[
                    "state/carry_distance_m"]!
                let intermediateFootUnloading = runtime.result.metrics[
                    "state/foot_unloading_fraction"]!
                for environment in parameters.indices {
                    let rootUp = intermediateStates[environment].humanoid
                        .root.rotation.act(F3(0, 0, 1)).z
                    let boxUp = intermediateStates[environment]
                        .manipulation.object.rotation
                        .act(F3(0, 0, 1)).z
                    let instantFeasible = intermediateFlags[environment]
                            .loadBearingGrasp
                        && intermediateFlags[environment].unsupported
                        && intermediateFlags[environment].physicallyLifted
                        && rootUp > 0.9 && boxUp > 0.9
                        && !intermediateFlags[environment].failed
                        && intermediateCarry[environment] >= configuration
                            .minimumTargetCarryDistanceMeters
                        && intermediateClearances[environment] >= configuration
                            .minimumTargetClearanceMeters
                        && intermediateFlags[environment].graspQuality
                            >= configuration.minimumTargetGraspQuality
                        && pathDownwardVelocityFeasible(
                            boxVerticalVelocity: intermediateStates[
                                environment].manipulation.object
                                .linearVelocity.z,
                            maximumDownwardVelocity: configuration
                                .maximumTargetPathDownwardBoxVelocityMPS)
                        && (configuration
                            .minimumTargetTerminalFootUnloadingFraction.map {
                                intermediateFootUnloading[environment] >= $0
                            } ?? true)
                    endpointDwells[environment] = instantFeasible
                        ? endpointDwells[environment] + 1 : 0
                    let instantWindowPenalty = [
                        hardMinimumPenalty(
                            value: intermediateFlags[environment]
                                .graspFrictionSupportFraction,
                            minimum: 1, scale: 0.25),
                        intermediateFlags[environment].unsupported ? 0 : 10,
                        intermediateFlags[environment].physicallyLifted
                            ? 0 : 10,
                        rootUp > 0.9 ? 0 : 10,
                        boxUp > 0.9 ? 0 : 10,
                        intermediateFlags[environment].failed ? 20 : 0,
                        hardMinimumPenalty(
                            value: intermediateCarry[environment],
                            minimum: configuration
                                .minimumTargetCarryDistanceMeters,
                            scale: 0.01),
                        hardMinimumPenalty(
                            value: intermediateClearances[environment],
                            minimum: configuration
                                .minimumTargetClearanceMeters,
                            scale: 0.01),
                        hardMinimumPenalty(
                            value: intermediateFlags[environment]
                                .graspQuality,
                            minimum: configuration
                                .minimumTargetGraspQuality,
                            scale: 0.05),
                        configuration
                            .minimumTargetTerminalFootUnloadingFraction.map {
                                hardMinimumPenalty(
                                    value: intermediateFootUnloading[
                                        environment],
                                    minimum: $0,
                                    scale: 0.10)
                            } ?? 0,
                        pathDownwardVelocityPenalty(
                            boxVerticalVelocity: intermediateStates[
                                environment].manipulation.object
                                .linearVelocity.z,
                            maximumDownwardVelocity: configuration
                                .maximumTargetPathDownwardBoxVelocityMPS),
                    ].max()!
                    appendFeasibilityPenalty(
                        instantWindowPenalty,
                        to: &endpointFeasibilityWindows[environment],
                        required:
                            configuration.targetFeasibilityDwellSteps)
                    if configuration.requireStableCarryPath {
                        let stable =
                            intermediateFlags[environment].loadBearingGrasp
                            && intermediateFlags[environment].unsupported
                            && intermediateFlags[environment].physicallyLifted
                            && rootUp > 0.9 && boxUp > 0.9
                            && intermediateClearances[environment] >= 0.01
                            && pathDownwardVelocityFeasible(
                                boxVerticalVelocity: intermediateStates[
                                    environment].manipulation.object
                                    .linearVelocity.z,
                                maximumDownwardVelocity: configuration
                                    .maximumTargetPathDownwardBoxVelocityMPS)
                            && !intermediateFlags[environment].failed
                        if !stable {
                            stablePathViolationSteps[environment] += 1
                        }
                        stableCarryPaths[environment] =
                            stableCarryPaths[environment] && stable
                    }
                }
            }
            simulatedEnvironmentControlSteps += parameters.count
                * (runtime.simulatedPreparationSteps + target.step)
            let terminals = states(runtime.task)
            let terminalFlags = flags(
                task: runtime.task, observation: runtime.observation,
                result: runtime.result)
            let carryDistances = runtime.result.metrics[
                "state/carry_distance_m"]!
            let clearances = runtime.result.metrics[
                "state/box_clearance_m"]!
            let terminalFootUnloading = runtime.result.metrics[
                "state/foot_unloading_fraction"]!
            return parameters.indices.map { index in
                Candidate(
                    parameters: parameters[index],
                    metrics: metrics(
                        from: terminals[index], to: target.state,
                        flags: terminalFlags[index],
                        carryDistance: carryDistances[index],
                        targetCarryDistance: target.carryDistance,
                        minimumCarryDistance:
                            configuration.minimumTargetCarryDistanceMeters,
                        clearance: clearances[index],
                        minimumClearance:
                            configuration.minimumTargetClearanceMeters,
                        graspQuality:
                            terminalFlags[index].graspQuality,
                        minimumGraspQuality:
                            configuration.minimumTargetGraspQuality,
                        feasibilityDwellSteps: endpointDwells[index],
                        requiredDwellSteps:
                            configuration.targetFeasibilityDwellSteps,
                        feasibilityWindowPenalty:
                            feasibilityWindowPenalty(
                                endpointFeasibilityWindows[index],
                                required: configuration
                                    .targetFeasibilityDwellSteps),
                        stablePathViolationSteps:
                            stablePathViolationSteps[index],
                        stablePathRequired:
                            configuration.requireStableCarryPath,
                        trajectorySteps: target.step,
                        terminalFootUnloadingFraction:
                            terminalFootUnloading[index],
                        minimumTerminalFootUnloadingFraction:
                            configuration
                                .minimumTargetTerminalFootUnloadingFraction),
                    terminal: terminals[index],
                    carryDistance: carryDistances[index],
                    warmupSteps: runtime.warmupSteps)
            }
        }

        // Replay the hidden target action only for an infrastructure audit. It
        // remains excluded from every search center, sample, and elite update.
        var targetReplayRuntime = try prepareSource(
            count: target.replicas.count)
        var targetReplayDwells = [Int](
            repeating: 0, count: target.replicas.count)
        var targetReplayFeasibilityWindows = [[Float]](
            repeating: [], count: target.replicas.count)
        var targetReplayPathViolations = [Int](
            repeating: 0, count: target.replicas.count)
        let targetSequenceBoundaryReplay =
            target.generatingTrajectorySequence.map {
                Self.structuredTrajectorySequenceBoundaryReplay(
                    $0,
                    phaseSteps:
                        target.generatingTrajectorySequencePhaseSteps,
                    denominator: configuration.recedingHorizonSteps,
                    initial: currentTrajectoryStart,
                    armKnotCount: configuration.trajectoryKnotCount,
                    blendKnotCount: configuration.legBlendKnotCount,
                    legResidualKnotCount:
                        configuration.legResidualKnotCount,
                    maximumLegResidualAction:
                        configuration.maximumLegResidualAction,
                    torsoResidualKnotCount:
                        configuration.torsoResidualKnotCount,
                    maximumTorsoResidualAction:
                        configuration.maximumTorsoResidualAction,
                    armAsymmetryKnotCount:
                        configuration.armAsymmetryKnotCount,
                    maximumArmAsymmetryAction:
                        configuration.maximumArmAsymmetryAction)
            }
        for step in 0..<target.step {
            let replayTrajectory = target.generatingTrajectorySequence.map {
                $0[step]
            } ?? selectedTargetTrajectory
            let actions = try applyTrajectory(
                [[Float]](repeating: replayTrajectory,
                          count: target.replicas.count),
                step: target.generatingTrajectorySequence == nil
                    ? step
                    : target.generatingTrajectorySequencePhaseSteps?[step]
                        ?? 0,
                denominator: target.generatingTrajectorySequence == nil
                    ? configuration.targetGenerationSteps
                    : configuration.recedingHorizonSteps,
                task: targetReplayRuntime.task,
                observation: targetReplayRuntime.observation,
                forwardOnlyBaseCommand:
                    target.generatingTrajectorySequence != nil
                        && configuration.recedingForwardOnlyBaseCommand,
                holonomicBaseCommand:
                    target.generatingTrajectorySequence != nil
                        && configuration.recedingHolonomicBaseCommand,
                locomotionCheckpointDirectory:
                    target.generatingTrajectorySequence != nil
                        ? configuration
                            .recedingLocomotionCheckpointDirectory
                        : nil,
                locomotionCommandSpeed:
                    target.generatingTrajectorySequence != nil
                        ? configuration.recedingLocomotionCommandSpeed
                        : nil,
                trajectoryStart: targetSequenceBoundaryReplay?.starts[step]
                    ?? currentTrajectoryStart,
                graspFeedback: target.graspFeedback)
            try targetReplayRuntime.task.step(
                actions: actions, into: &targetReplayRuntime.result)
            targetReplayRuntime.observation =
                targetReplayRuntime.result.observations
            let replayStates = states(targetReplayRuntime.task)
            let replayFlags = flags(
                task: targetReplayRuntime.task,
                observation: targetReplayRuntime.observation,
                result: targetReplayRuntime.result)
            let replayCarry = targetReplayRuntime.result.metrics[
                "state/carry_distance_m"]!
            let replayClearance = targetReplayRuntime.result.metrics[
                "state/box_clearance_m"]!
            let replayFootUnloading = targetReplayRuntime.result.metrics[
                "state/foot_unloading_fraction"]!
            for environment in targetReplayDwells.indices {
                let rootUp = replayStates[environment].humanoid.root
                    .rotation.act(F3(0, 0, 1)).z
                let boxUp = replayStates[environment].manipulation.object
                    .rotation.act(F3(0, 0, 1)).z
                let feasible = replayFlags[environment].loadBearingGrasp
                    && replayFlags[environment].unsupported
                    && replayFlags[environment].physicallyLifted
                    && rootUp > 0.9 && boxUp > 0.9
                    && !replayFlags[environment].failed
                    && replayCarry[environment]
                        >= configuration.minimumTargetCarryDistanceMeters
                    && replayClearance[environment]
                        >= configuration.minimumTargetClearanceMeters
                    && replayFlags[environment].graspQuality
                        >= configuration.minimumTargetGraspQuality
                    && pathDownwardVelocityFeasible(
                        boxVerticalVelocity: replayStates[environment]
                            .manipulation.object.linearVelocity.z,
                        maximumDownwardVelocity: configuration
                            .maximumTargetPathDownwardBoxVelocityMPS)
                    && (configuration
                        .minimumTargetTerminalFootUnloadingFraction.map {
                            replayFootUnloading[environment] >= $0
                        } ?? true)
                targetReplayDwells[environment] = feasible
                    ? targetReplayDwells[environment] + 1 : 0
                let instantWindowPenalty = [
                    hardMinimumPenalty(
                        value: replayFlags[environment]
                            .graspFrictionSupportFraction,
                        minimum: 1, scale: 0.25),
                    replayFlags[environment].unsupported ? 0 : 10,
                    replayFlags[environment].physicallyLifted ? 0 : 10,
                    rootUp > 0.9 ? 0 : 10,
                    boxUp > 0.9 ? 0 : 10,
                    replayFlags[environment].failed ? 20 : 0,
                    hardMinimumPenalty(
                        value: replayCarry[environment],
                        minimum:
                            configuration.minimumTargetCarryDistanceMeters,
                        scale: 0.01),
                    hardMinimumPenalty(
                        value: replayClearance[environment],
                        minimum:
                            configuration.minimumTargetClearanceMeters,
                        scale: 0.01),
                    hardMinimumPenalty(
                        value: replayFlags[environment].graspQuality,
                        minimum: configuration.minimumTargetGraspQuality,
                        scale: 0.05),
                    configuration
                        .minimumTargetTerminalFootUnloadingFraction.map {
                            hardMinimumPenalty(
                                value: replayFootUnloading[environment],
                                minimum: $0,
                                scale: 0.10)
                        } ?? 0,
                    pathDownwardVelocityPenalty(
                        boxVerticalVelocity: replayStates[environment]
                            .manipulation.object.linearVelocity.z,
                        maximumDownwardVelocity: configuration
                            .maximumTargetPathDownwardBoxVelocityMPS),
                ].max()!
                appendFeasibilityPenalty(
                    instantWindowPenalty,
                    to: &targetReplayFeasibilityWindows[environment],
                    required: configuration.targetFeasibilityDwellSteps)
                if configuration.requireStableCarryPath {
                    let stable =
                        replayFlags[environment].loadBearingGrasp
                        && replayFlags[environment].unsupported
                        && replayFlags[environment].physicallyLifted
                        && rootUp > 0.9 && boxUp > 0.9
                        && replayClearance[environment] >= 0.01
                        && pathDownwardVelocityFeasible(
                            boxVerticalVelocity: replayStates[environment]
                                .manipulation.object.linearVelocity.z,
                            maximumDownwardVelocity: configuration
                                .maximumTargetPathDownwardBoxVelocityMPS)
                        && !replayFlags[environment].failed
                    if !stable { targetReplayPathViolations[environment] += 1 }
                }
            }
        }
        simulatedEnvironmentControlSteps += target.replicas.count
            * (targetReplayRuntime.simulatedPreparationSteps + target.step)
        let targetReplayStates = states(targetReplayRuntime.task)
        let targetReplayFlags = flags(
            task: targetReplayRuntime.task,
            observation: targetReplayRuntime.observation,
            result: targetReplayRuntime.result)
        let targetReplayCarryDistances = targetReplayRuntime.result.metrics[
            "state/carry_distance_m"]!
        let targetReplayClearances = targetReplayRuntime.result.metrics[
            "state/box_clearance_m"]!
        let targetReplayFootUnloading =
            targetReplayRuntime.result.metrics[
                "state/foot_unloading_fraction"]!
        let targetReplayMetrics = metrics(
            from: targetReplayStates[0], to: target.state,
            flags: targetReplayFlags[0],
            carryDistance: targetReplayCarryDistances[0],
            targetCarryDistance: target.carryDistance,
            minimumCarryDistance:
                configuration.minimumTargetCarryDistanceMeters,
            clearance: targetReplayClearances[0],
            minimumClearance:
                configuration.minimumTargetClearanceMeters,
            graspQuality: targetReplayFlags[0].graspQuality,
            minimumGraspQuality:
                configuration.minimumTargetGraspQuality,
            feasibilityDwellSteps: targetReplayDwells[0],
            requiredDwellSteps:
                configuration.targetFeasibilityDwellSteps,
            feasibilityWindowPenalty: feasibilityWindowPenalty(
                targetReplayFeasibilityWindows[0],
                required: configuration.targetFeasibilityDwellSteps),
            stablePathViolationSteps: targetReplayPathViolations[0],
            stablePathRequired: configuration.requireStableCarryPath,
            trajectorySteps: target.step,
            terminalFootUnloadingFraction:
                targetReplayFootUnloading[0],
            minimumTerminalFootUnloadingFraction:
                configuration.minimumTargetTerminalFootUnloadingFraction)
        let targetCloneMetrics = target.replicas.indices.map { index in
            metrics(
                from: target.replicas[index], to: target.state,
                flags: target.replicaFlags[index],
                carryDistance: target.replicaCarryDistances[index],
                targetCarryDistance: target.carryDistance,
                minimumCarryDistance:
                    configuration.minimumTargetCarryDistanceMeters,
                clearance: target.replicaClearances[index],
                minimumClearance:
                    configuration.minimumTargetClearanceMeters,
                graspQuality: target.replicaGraspQualities[index],
                minimumGraspQuality:
                    configuration.minimumTargetGraspQuality,
                feasibilityDwellSteps:
                    configuration.targetFeasibilityDwellSteps,
                requiredDwellSteps:
                    configuration.targetFeasibilityDwellSteps,
                stablePathViolationSteps: 0,
                stablePathRequired: configuration.requireStableCarryPath,
                trajectorySteps: target.step,
                terminalFootUnloadingFraction:
                    target.replicaFootUnloadingFractions[index],
                minimumTerminalFootUnloadingFraction:
                    configuration.minimumTargetTerminalFootUnloadingFraction)
        }
        let targetCloneSuccessFraction = Float(targetCloneMetrics.filter {
            $0.endpointPassed
        }.count) / Float(targetCloneMetrics.count)

        var generator = ProbeRandomNumberGenerator(
            seed: optimizerSeed &+ 0xF10A)
        let zero = [Float](repeating: 0, count: parameterCount)
        var searchMean = proposalTrajectory ?? zero
        var covariance = [[Float]](
            repeating: [Float](repeating: 0, count: parameterCount),
            count: parameterCount)
        for index in 0..<parameterCount {
            covariance[index][index] = configuration.initialStandardDeviation
                * configuration.initialStandardDeviation
        }
        var overallBest: ValidatedCandidate?
        var overallNearMiss: ValidatedCandidate?
        var initialProvidedMetrics: HumanoidBoxPhysicalFlowMetrics?
        var zeroMetrics: HumanoidBoxPhysicalFlowMetrics?
        var selectedProposal: PhysicalFlowProposalSelection =
            proposalTrajectory == nil ? .notApplicable : .provided
        var providedProbeBestLoss: Float?
        var zeroProbeBestLoss: Float?
        var histories = [HumanoidBoxPhysicalFlowGeneration]()
        var candidateRollouts = 0
        var optimizationEvaluationIndex: UInt64 = 0

        func evaluateBatchedForSearch(
            _ parameters: [[Float]]
        ) throws -> [Candidate] {
            let noise = configuration
                .optimizationActionNoiseStandardDeviation
            let replicas = noise > 0
                ? configuration.optimizationActionNoiseReplicaCount : 1
            let expanded = parameters.flatMap { parameter in
                [[Float]](repeating: parameter, count: replicas)
            }
            optimizationEvaluationIndex &+= 1
            let evaluated = try evaluate(
                expanded,
                actionNoiseStandardDeviation: noise,
                actionNoiseSeedOffset: optimizationEvaluationIndex)
            candidateRollouts += evaluated.count
            guard replicas > 1 else { return evaluated }
            return parameters.indices.map { index in
                let start = index * replicas
                let end = start + replicas
                // Minimize the worst perturbed rollout. This avoids selecting
                // a trajectory that succeeds only under a favorable noise draw.
                return evaluated[start..<end].max {
                    $0.metrics.loss < $1.metrics.loss
                }!
            }
        }

        func evaluateForSearch(
            _ parameters: [[Float]]
        ) throws -> [Candidate] {
            guard configuration.reconstructionOneEnvironmentSearch else {
                return try evaluateBatchedForSearch(parameters)
            }

            let noise = configuration
                .optimizationActionNoiseStandardDeviation
            let replicas = noise > 0
                ? configuration.optimizationActionNoiseReplicaCount : 1
            var selected = [Candidate]()
            selected.reserveCapacity(parameters.count)
            for parameter in parameters {
                var evaluated = [Candidate]()
                evaluated.reserveCapacity(replicas)
                for _ in 0..<replicas {
                    optimizationEvaluationIndex &+= 1
                    evaluated.append(try evaluate(
                        [parameter],
                        actionNoiseStandardDeviation: noise,
                        actionNoiseSeedOffset:
                            optimizationEvaluationIndex)[0])
                    candidateRollouts += 1
                }
                // Preserve the robust-search interpretation used by batched
                // optimization: a proposal is ranked by its worst perturbed
                // rollout. With no action noise this is its exact UI-layout
                // reconstruction loss.
                selected.append(evaluated.max {
                    $0.metrics.loss < $1.metrics.loss
                }!)
            }
            return selected
        }

        func sample(around center: [Float], transform: [[Float]]) -> [Float] {
            let noise = center.map { _ in generator.normal() }
            return center.indices.map { row in
                let delta = (0...row).reduce(Float(0)) {
                    $0 + transform[row][$1] * noise[$1]
                }
                return simd_clamp(center[row] + delta, -0.999, 0.999)
            }
        }

        for generation in 0..<configuration.generations {
            let transform = cholesky(covariance)
            let candidates: [Candidate]
            if generation == 0, let proposalTrajectory {
                let maximumProbe = min(
                    configuration.proposalProbeSize,
                    configuration.populationSize)
                let probeCount = max(2, maximumProbe - maximumProbe % 2)
                var modes = [PhysicalFlowProposalSelection]()
                let probeParameters = (0..<probeCount).map { index -> [Float] in
                    let mode: PhysicalFlowProposalSelection =
                        index.isMultiple(of: 2) ? .provided : .geometric
                    modes.append(mode)
                    let center = mode == .provided ? proposalTrajectory : zero
                    return index < 2 ? center
                        : sample(around: center, transform: transform)
                }
                let probe = try evaluateForSearch(probeParameters)
                initialProvidedMetrics = probe[0].metrics
                zeroMetrics = probe[1].metrics
                var bestProvided = probe[0]
                var bestZero = probe[1]
                for index in probe.indices {
                    if modes[index] == .provided,
                       probe[index].metrics.loss < bestProvided.metrics.loss {
                        bestProvided = probe[index]
                    } else if modes[index] == .geometric,
                              probe[index].metrics.loss
                                < bestZero.metrics.loss {
                        bestZero = probe[index]
                    }
                }
                providedProbeBestLoss = bestProvided.metrics.loss
                zeroProbeBestLoss = bestZero.metrics.loss
                let winner: Candidate
                if bestProvided.metrics.loss <= bestZero.metrics.loss {
                    selectedProposal = .provided
                    winner = bestProvided
                } else {
                    selectedProposal = .geometric
                    winner = bestZero
                }
                let remaining = configuration.populationSize - probeCount
                if remaining > 0 {
                    let exploitation = (0..<remaining).map { _ in
                        sample(around: winner.parameters, transform: transform)
                    }
                    let evaluated = try evaluateForSearch(exploitation)
                    candidates = probe + evaluated
                } else {
                    candidates = probe
                }
            } else {
                var parameters = [[Float]]()
                parameters.reserveCapacity(configuration.populationSize)
                for index in 0..<configuration.populationSize {
                    if index == 0 {
                        parameters.append(searchMean)
                    } else if index == 1, let overallBest {
                        parameters.append(overallBest.candidate.parameters)
                    } else {
                        parameters.append(sample(
                            around: searchMean, transform: transform))
                    }
                }
                candidates = try evaluateForSearch(parameters)
                if generation == 0 {
                    zeroMetrics = candidates[0].metrics
                }
            }
            guard candidates.allSatisfy({ $0.metrics.loss.isFinite }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite humanoid-box physical-flow candidate")
            }
            let sorted = candidates.sorted {
                $0.metrics.loss < $1.metrics.loss
            }
            let validated = try sorted.prefix(
                configuration.reconstructionValidationCandidateCount
            ).map { proposal -> ValidatedCandidate in
                // The one-environment solver used by Policy Replay has a
                // different global body/contact layout from row zero of a
                // batched scene. Require both execution regimes explicitly.
                let exact = try evaluate([proposal.parameters])[0]
                candidateRollouts += 1
                let repeated = [[Float]](
                    repeating: proposal.parameters,
                    count: configuration.populationSize)
                // Acceptance remains batched even when proposal ranking is
                // deliberately performed in the one-environment layout.
                let replicas = try evaluateBatchedForSearch(repeated)
                let summary = robustValidationSummary(
                    replicaPasses: replicas.map {
                        $0.metrics.endpointPassed
                    },
                    minimumSuccessFraction: configuration
                        .reconstructionValidationMinimumSuccessFraction)
                let batchQuantileLoss = replicas.map {
                    $0.metrics.loss
                }.sorted()[summary.upperQuantileIndex]
                return ValidatedCandidate(
                    candidate: exact,
                    robustLoss: max(
                        exact.metrics.loss, batchQuantileLoss),
                    successFraction: summary.successFraction)
            }.sorted { $0.robustLoss < $1.robustLoss }
            if let near = validated.first,
               overallNearMiss == nil
                || near.robustLoss < overallNearMiss!.robustLoss {
                overallNearMiss = near
            }
            if let accepted = validated.first(where: {
                $0.candidate.metrics.endpointPassed
                    && $0.successFraction >= configuration
                        .reconstructionValidationMinimumSuccessFraction
            }), overallBest == nil
                || accepted.robustLoss < overallBest!.robustLoss {
                overallBest = accepted
            }
            let eliteCount = reconstructionEliteCount(
                populationSize: configuration.populationSize,
                eliteFraction: configuration.eliteFraction,
                validatedCandidateCount: validated.count)
            // CEM must learn from the execution regimes used by the final
            // certificate. Updating from the original batched ranking here
            // lets a batch-only contact solution dominate forever even when
            // every dedicated one-environment replay rejects it.
            let elites = validated.prefix(
                min(eliteCount, validated.count)).map(\.candidate)
            precondition(elites.count >= 2)
            var nextMean = [Float](repeating: 0, count: parameterCount)
            for elite in elites {
                for index in 0..<parameterCount {
                    nextMean[index] += elite.parameters[index]
                }
            }
            for index in 0..<parameterCount {
                nextMean[index] /= Float(eliteCount)
            }
            var nextCovariance = [[Float]](
                repeating: [Float](repeating: 0, count: parameterCount),
                count: parameterCount)
            for elite in elites {
                for row in 0..<parameterCount {
                    for column in 0..<parameterCount {
                        nextCovariance[row][column] +=
                            (elite.parameters[row] - nextMean[row])
                            * (elite.parameters[column] - nextMean[column])
                    }
                }
            }
            for row in 0..<parameterCount {
                for column in 0..<parameterCount {
                    nextCovariance[row][column] /= Float(eliteCount)
                }
                nextCovariance[row][row] = max(
                    nextCovariance[row][row], 0.02 * 0.02)
            }
            searchMean = zip(searchMean, nextMean).map {
                0.25 * $0.0 + 0.75 * $0.1
            }
            for row in 0..<parameterCount {
                for column in 0..<parameterCount {
                    covariance[row][column] = 0.25
                        * covariance[row][column]
                        + 0.75 * nextCovariance[row][column]
                }
            }
            let losses = candidates.map(\.metrics.loss).sorted()
            histories.append(.init(
                generation: generation,
                bestLoss: validated[0].robustLoss,
                medianLoss: losses[losses.count / 2],
                bestMaximumNormalizedError:
                    validated[0].candidate.metrics.maximumNormalizedError,
                meanStandardDeviation: (0..<parameterCount).reduce(0) {
                    $0 + sqrt(max(covariance[$1][$1], 0))
                } / Float(parameterCount),
                warmupControlSteps:
                    validated[0].candidate.warmupSteps))
        }

        let selectedValidation = overallBest ?? overallNearMiss!
        let best = selectedValidation.candidate
        let exactSelected = try evaluate([best.parameters])[0]
        let exactRepeated = try evaluate([best.parameters])[0]
        let robustParameters = [[Float]](
            repeating: best.parameters,
            count: configuration.robustReplayCount)
        let robustCandidates = try evaluate(
            robustParameters,
            actionNoiseStandardDeviation:
                configuration.robustActionNoiseStandardDeviation,
            actionNoiseSeedOffset: 0xB057_0001)
        let robustErrors = robustCandidates.map {
            $0.metrics.maximumNormalizedError
        }.sorted()
        let robustSuccessFraction = Float(robustCandidates.filter {
            $0.metrics.endpointPassed
        }.count) / Float(robustCandidates.count)
        let selectedReplayStateError = stateEndpointEvaluation(
            from: exactRepeated.terminal, to: exactSelected.terminal)
            .maximumNormalizedError
        let infrastructureGatePassed =
            targetReplayMetrics.maximumNormalizedError < 0.02
                && selectedReplayStateError < 0.02
        let targetGatePassed = target.flags.loadBearingGrasp
            && target.flags.unsupported && target.flags.physicallyLifted
            && target.flags.robotUpright && target.flags.boxUpright
            && target.robotUpright > 0.9 && target.boxUpright > 0.9
            && target.clearance
                >= configuration.minimumTargetClearanceMeters
            && target.carryDistance
                >= configuration.minimumTargetCarryDistanceMeters
            && target.flags.graspQuality
                >= configuration.minimumTargetGraspQuality
            && minimumDestinationProgressPassed(
                target.destinationProgress,
                minimum:
                    configuration.minimumTargetDestinationProgressMeters)
            && target.rootDestinationProgress
                >= configuration.minimumTargetRootDestinationProgressMeters
            && target.loadedTouchdowns
                >= configuration.minimumTargetTouchdowns
            && target.loadedAlternatingSteps
                >= configuration.minimumTargetAlternatingSteps
            && target.maximumSwingFootLift
                >= configuration.minimumTargetSwingFootLiftMeters
            && target.maximumLoadedFootAirTime
                >= configuration.minimumTargetFootAirTimeSeconds
            && target.maximumFootUnloadingFraction
                >= configuration.minimumTargetFootUnloadingFraction
            && (configuration
                .minimumTargetTerminalFootUnloadingFraction.map {
                    target.terminalFootUnloadingFraction >= $0
                } ?? true)
            && (!configuration.requireStableCarryPath
                || target.stableCarryPath)
        let reconstructionGatePassed = exactSelected.metrics.endpointPassed
        let robustReplayGatePassed = robustSuccessFraction >= 0.8
        let targetPlanningGatePassed = targetGatePassed
            && targetCloneSuccessFraction >= 0.8
            && targetReplayMetrics.maximumNormalizedError < 0.02
        let targetReusableFrontierGatePassed = targetPlanningGatePassed
            && (configuration.recedingHorizonSteps == 0
                || target.predictedRecoveryPathSafe)
        let zeroLoss = max(zeroMetrics!.loss, 1e-12)
        let targetTrajectoryWasWithheld = selectedTargetTrajectory != zero
            && selectedTargetTrajectory != proposalTrajectory

        return HumanoidBoxPhysicalFlowReport(
            experiment: "humanoid-box-state-to-state-physical-flow-v0",
            checkpointDirectory: checkpointDirectory,
            seed: configuration.seed,
            optimizerSeed: configuration.optimizerSeed,
            populationSize: configuration.populationSize,
            proposalProbeSize: configuration.proposalProbeSize,
            generations: configuration.generations,
            targetTrajectoryWithheldFromSearch:
                targetTrajectoryWasWithheld,
            targetGenerationSteps: configuration.targetGenerationSteps,
            targetExecutionSteps: targetExecutionSteps,
            candidateTrajectoryDurationSteps:
                configuration.candidateTrajectoryDurationSteps,
            targetDiscoveryPopulationSize:
                configuration.targetDiscoveryPopulationSize,
            targetDiscoveryGenerations:
                configuration.targetDiscoveryGenerations,
            targetDiscoveryInitialStandardDeviation:
                configuration.targetDiscoveryInitialStandardDeviation,
            targetDiscoveryCandidateRollouts:
                targetDiscoveryCandidateRollouts,
            targetDiscoveryArmOnly:
                configuration.targetDiscoveryArmOnly ? true : nil,
            targetDiscoveryLowerBodyOnly:
                configuration.targetDiscoveryLowerBodyOnly ? true : nil,
            targetDiscoveryTiedArmKnots:
                configuration.targetDiscoveryTiedArmKnots ? true : nil,
            targetDiscoveryBlockCoordinateSearch:
                configuration.targetDiscoveryBlockCoordinateSearch
                    ? true : nil,
            targetDiscoverySeededFromSourceTerminalUpperBody:
                seedsFromSourceTerminalTrajectory ? true : nil,
            targetDiscoveryHeldSourceTerminalUpperBody:
                holdsSourceTerminalUpperBody ? true : nil,
            targetDiscoveryPreservedProvidedUpperBodySeed:
                preservesProvidedUpperBodySeed ? true : nil,
            targetGeneratingTrajectory: selectedTargetTrajectory,
            targetGeneratingTrajectorySequence:
                target.generatingTrajectorySequence,
            targetGeneratingTrajectorySequencePhaseSteps:
                target.generatingTrajectorySequencePhaseSteps,
            derivedStageContinuesFromSourceTerminal:
                configuration.continueTrajectoryFromSourceTerminal
                    ? true : nil,
            targetCommittedTrace: target.committedTrace,
            recedingHorizonSteps: configuration.recedingHorizonSteps > 0
                ? configuration.recedingHorizonSteps : nil,
            recedingInitialPhaseStep:
                configuration.recedingInitialPhaseStep > 0
                    ? configuration.recedingInitialPhaseStep : nil,
            recedingControlHorizonSteps:
                configuration.recedingHorizonSteps > 0
                    ? configuration.recedingControlHorizonSteps : nil,
            recedingSafetyLookaheadSteps:
                configuration.recedingSafetyLookaheadSteps,
            recedingTerminalHoldSteps:
                configuration.recedingTerminalHoldSteps > 0
                    ? configuration.recedingTerminalHoldSteps : nil,
            recedingValidationCandidateCount:
                configuration.recedingHorizonSteps > 0
                    ? configuration.recedingValidationCandidateCount : nil,
            recedingValidationReplicaCount:
                configuration.recedingHorizonSteps > 0
                    ? configuration.recedingValidationReplicaCount : nil,
            recedingValidationMinimumSuccessFraction:
                configuration.recedingHorizonSteps > 0
                    ? configuration
                        .recedingValidationMinimumSuccessFraction : nil,
            reconstructionValidationCandidateCount:
                configuration.reconstructionValidationCandidateCount,
            reconstructionValidationMinimumSuccessFraction:
                configuration
                    .reconstructionValidationMinimumSuccessFraction,
            reconstructionValidationSuccessFraction:
                selectedValidation.successFraction,
            reconstructionOneEnvironmentSearch:
                configuration.reconstructionOneEnvironmentSearch,
            recedingShiftInitialSeed:
                configuration.recedingShiftInitialSeed ? true : nil,
            recedingLocomotionBlendProposal:
                configuration.recedingLocomotionBlendProposal,
            recedingLocomotionZeroResidualProposal:
                configuration.recedingLocomotionZeroResidualProposal
                    ? true : nil,
            recedingLocomotionCheckpointDirectory: configuration
                .recedingLocomotionCheckpointDirectory,
            recedingLocomotionCommandSpeed: configuration
                .recedingLocomotionCommandSpeed,
            recedingForwardOnlyBaseCommand:
                configuration.recedingForwardOnlyBaseCommand,
            recedingHolonomicBaseCommand:
                configuration.recedingHolonomicBaseCommand,
            carryBaseLegActionFractionOverride:
                configuration.carryBaseLegActionFractionOverride,
            legBlendKnotCount: configuration.legBlendKnotCount,
            legResidualKnotCount: configuration.legResidualKnotCount,
            maximumLegResidualAction:
                configuration.maximumLegResidualAction,
            torsoResidualKnotCount:
                configuration.torsoResidualKnotCount,
            maximumTorsoResidualAction:
                configuration.maximumTorsoResidualAction,
            armAsymmetryKnotCount:
                configuration.armAsymmetryKnotCount,
            maximumArmAsymmetryAction:
                configuration.maximumArmAsymmetryAction,
            graspAnchorFeedbackBlend:
                target.graspFeedback?.blend,
            graspAnchorFeedbackVelocityHorizonSeconds:
                target.graspFeedback?.velocityHorizonSeconds,
            graspAnchorFeedbackMaximumActionCorrection:
                target.graspFeedback?.maximumActionCorrection,
            graspAnchorFeedbackInwardPreloadMeters:
                target.graspFeedback?.inwardPreloadMeters,
            leftGraspAnchorBoxLocalMeters:
                target.graspFeedback.map {
                    [$0.leftAnchorBoxLocal.x, $0.leftAnchorBoxLocal.y,
                     $0.leftAnchorBoxLocal.z]
                },
            rightGraspAnchorBoxLocalMeters:
                target.graspFeedback.map {
                    [$0.rightAnchorBoxLocal.x, $0.rightAnchorBoxLocal.y,
                     $0.rightAnchorBoxLocal.z]
                },
            graspAnchorBoxHeightMeters:
                target.graspFeedback?.boxHeightMeters,
            minimumTargetCarryDistanceMeters:
                configuration.minimumTargetCarryDistanceMeters,
            minimumTargetPathCarryDistanceMeters:
                minimumTargetPathCarryDistance,
            targetDiscoveryObjectiveCarryDistanceMeters: configuration
                .targetDiscoveryObjectiveCarryDistanceMeters
                ?? configuration.minimumTargetCarryDistanceMeters,
            minimumTargetDestinationProgressMeters: configuration
                .minimumTargetDestinationProgressMeters,
            targetDiscoveryObjectiveDestinationProgressMeters:
                targetSelectionDestinationProgress,
            minimumTargetRootDestinationProgressMeters: configuration
                .minimumTargetRootDestinationProgressMeters,
            targetDiscoveryObjectiveRootDestinationProgressMeters:
                targetSelectionRootDestinationProgress,
            minimumTargetTouchdowns:
                configuration.minimumTargetTouchdowns,
            minimumTargetAlternatingSteps:
                configuration.minimumTargetAlternatingSteps,
            minimumTargetSwingFootLiftMeters:
                configuration.minimumTargetSwingFootLiftMeters,
            targetDiscoveryObjectiveSwingFootLiftMeters:
                configuration.targetDiscoveryObjectiveSwingFootLiftMeters,
            minimumTargetFootAirTimeSeconds:
                configuration.minimumTargetFootAirTimeSeconds,
            targetDiscoveryObjectiveFootAirTimeSeconds:
                configuration.targetDiscoveryObjectiveFootAirTimeSeconds,
            minimumTargetFootUnloadingFraction:
                configuration.minimumTargetFootUnloadingFraction,
            targetDiscoveryObjectiveFootUnloadingFraction:
                configuration
                    .targetDiscoveryObjectiveFootUnloadingFraction,
            minimumTargetTerminalFootUnloadingFraction:
                configuration
                    .minimumTargetTerminalFootUnloadingFraction,
            minimumTargetClearanceMeters:
                configuration.minimumTargetClearanceMeters,
            targetDiscoveryObjectiveClearanceMeters:
                targetDiscoveryClearance,
            maximumTargetPathDownwardBoxVelocityMPS:
                configuration
                    .maximumTargetPathDownwardBoxVelocityMPS,
            minimumTargetTerminalClearanceMeters:
                configuration.minimumTargetTerminalClearanceMeters,
            maximumTargetTerminalDownwardBoxVelocityMPS:
                configuration
                    .maximumTargetTerminalDownwardBoxVelocityMPS,
            minimumTargetGraspQuality:
                configuration.minimumTargetGraspQuality,
            targetDiscoveryObjectiveGraspQuality:
                targetDiscoveryGraspQuality,
            targetFeasibilityDwellSteps:
                configuration.targetFeasibilityDwellSteps,
            requireStableCarryPath: configuration.requireStableCarryPath,
            sourceTrajectorySteps: sourceControlSteps,
            sourceStages: sourceStages,
            sourceWarmupAppliedActions: sourceWarmupAppliedActions,
            sourceAppliedActions: sourceAppliedActions,
            sourceReplaySuccessFraction:
                target.sourceReplaySuccessFraction,
            selectedTargetStep: target.step,
            targetClearanceMeters: target.clearance,
            targetGraspQuality: target.flags.graspQuality,
            targetBoxUprightAlignment: target.boxUpright,
            targetRobotUprightAlignment: target.robotUpright,
            targetCarryDistanceMeters: target.carryDistance,
            targetPlacementDistanceMeters: target.placementDistance,
            targetDestinationProgressMeters: target.destinationProgress,
            targetRootDestinationProgressMeters:
                target.rootDestinationProgress,
            targetLoadedTouchdowns:
                target.loadedTouchdowns,
            targetLoadedAlternatingSteps:
                target.loadedAlternatingSteps,
            targetMaximumSwingFootLiftMeters:
                target.maximumSwingFootLift,
            targetMaximumLoadedFootAirTimeSeconds:
                target.maximumLoadedFootAirTime,
            targetMaximumFootUnloadingFraction:
                target.maximumFootUnloadingFraction,
            targetTerminalFootUnloadingFraction:
                target.terminalFootUnloadingFraction,
            targetStableCarryPath: target.stableCarryPath,
            targetPredictedRecoveryPathSafe:
                configuration.recedingHorizonSteps > 0
                    ? target.predictedRecoveryPathSafe : nil,
            targetCloneSuccessFraction: targetCloneSuccessFraction,
            targetReplayMaximumNormalizedError:
                targetReplayMetrics.maximumNormalizedError,
            providedProposal: initialProvidedMetrics,
            zeroProposal: zeroMetrics!,
            generationZeroSelectedProposal: selectedProposal,
            providedProposalProbeBestLoss: providedProbeBestLoss,
            zeroProposalProbeBestLoss: zeroProbeBestLoss,
            optimized: exactSelected.metrics,
            optimizedToZeroLossRatio: exactSelected.metrics.loss / zeroLoss,
            robustReplaySuccessFraction: robustSuccessFraction,
            robustActionNoiseStandardDeviation:
                configuration.robustActionNoiseStandardDeviation,
            optimizationActionNoiseStandardDeviation:
                configuration.optimizationActionNoiseStandardDeviation,
            optimizationActionNoiseReplicaCount:
                configuration.optimizationActionNoiseReplicaCount,
            initialStandardDeviation:
                configuration.initialStandardDeviation,
            eliteFraction: configuration.eliteFraction,
            robustReplayMedianMaximumNormalizedError:
                robustErrors[robustErrors.count / 2],
            robustReplayWorstMaximumNormalizedError: robustErrors.last!,
            selectedReplayMaximumNormalizedStateError:
                selectedReplayStateError,
            bestTrajectory: best.parameters,
            generationHistory: histories,
            candidateRollouts: candidateRollouts
                + configuration.robustReplayCount + 2,
            simulatedEnvironmentControlSteps:
                simulatedEnvironmentControlSteps,
            elapsedSeconds: Date().timeIntervalSince(startTime),
            infrastructureGatePassed: infrastructureGatePassed,
            targetGatePassed: targetGatePassed,
            targetFinitePrefixGatePassed: targetPlanningGatePassed,
            targetPlanningGatePassed: targetPlanningGatePassed,
            targetReusableFrontierGatePassed:
                targetReusableFrontierGatePassed,
            reconstructionGatePassed: reconstructionGatePassed,
            robustReplayGatePassed: robustReplayGatePassed,
            goGatePassed: infrastructureGatePassed
                && targetReusableFrontierGatePassed
                && reconstructionGatePassed && robustReplayGatePassed)
    }

    static func legBlendFraction(
        _ parameters: [Float], progress: Float,
        armParameterCount: Int, knotCount: Int
    ) -> Float {
        precondition(knotCount > 0)
        precondition(parameters.count >= armParameterCount + knotCount)
        let scaled = simd_clamp(progress, 0, 1) * Float(knotCount)
        let lower = min(Int(floor(scaled)), knotCount - 1)
        let fraction = scaled - Float(lower)
        func knot(_ index: Int) -> Float {
            guard index > 0 else { return 0 }
            return simd_clamp(
                parameters[armParameterCount + index - 1], 0, 1)
        }
        return (1 - fraction) * knot(lower)
            + fraction * knot(min(lower + 1, knotCount))
    }

    /// Advance a receding-horizon spline by one control interval. Without
    /// this shift, every MPC iteration restarts the selected spline at its
    /// first knot and silently discards the viable future it just simulated.
    /// Parameters remain in their normalized representation so the shifted
    /// trajectory is a valid CEM mean, not an injected action.
    static func shiftedRecedingTrajectory(
        _ parameters: [Float], horizon: Int,
        armKnotCount: Int, blendKnotCount: Int,
        legResidualKnotCount: Int, maximumLegResidualAction: Float,
        torsoResidualKnotCount: Int,
        maximumTorsoResidualAction: Float,
        armAsymmetryKnotCount: Int,
        maximumArmAsymmetryAction: Float
    ) -> [Float] {
        precondition(horizon > 0 && armKnotCount > 0)
        let armParameterCount = 4 * armKnotCount
        let expected = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount + torsoResidualKnotCount
            + 4 * armAsymmetryKnotCount
        precondition(parameters.count == expected)
        let shift = 1 / Float(horizon)
        var shifted = parameters
        func sourceProgress(knot: Int, count: Int) -> Float {
            min(Float(knot) / Float(count) + shift, 1)
        }
        for knot in 1...armKnotCount {
            let sampled = HumanoidBoxCarryActuationProbe
                .trajectoryArmDelta(
                    Array(parameters.prefix(armParameterCount)),
                    knotCount: armKnotCount,
                    progress: sourceProgress(
                        knot: knot, count: armKnotCount))
            let offset = 4 * (knot - 1)
            for component in 0..<4 {
                shifted[offset + component] = simd_clamp(
                    sampled[component], -0.999, 0.999)
            }
        }
        if blendKnotCount > 0 {
            for knot in 1...blendKnotCount {
                shifted[armParameterCount + knot - 1] = legBlendFraction(
                    parameters,
                    progress: sourceProgress(
                        knot: knot, count: blendKnotCount),
                    armParameterCount: armParameterCount,
                    knotCount: blendKnotCount)
            }
        }
        let legOffset = armParameterCount + blendKnotCount
        if legResidualKnotCount > 0 {
            for knot in 1...legResidualKnotCount {
                let progress = sourceProgress(
                    knot: knot, count: legResidualKnotCount)
                for action in 0..<10 {
                    shifted[legOffset + (knot - 1) * 10 + action] =
                        simd_clamp(
                            legResidualAction(
                                parameters, action: action,
                                progress: progress,
                                armParameterCount: armParameterCount,
                                blendKnotCount: blendKnotCount,
                                residualKnotCount: legResidualKnotCount,
                                maximumAction: maximumLegResidualAction)
                                / maximumLegResidualAction,
                            -0.999, 0.999)
                }
            }
        }
        let torsoOffset = legOffset + 10 * legResidualKnotCount
        if torsoResidualKnotCount > 0 {
            for knot in 1...torsoResidualKnotCount {
                shifted[torsoOffset + knot - 1] = simd_clamp(
                    torsoResidualAction(
                        parameters,
                        progress: sourceProgress(
                            knot: knot, count: torsoResidualKnotCount),
                        armParameterCount: armParameterCount,
                        blendKnotCount: blendKnotCount,
                        legResidualKnotCount: legResidualKnotCount,
                        torsoResidualKnotCount: torsoResidualKnotCount,
                        maximumAction: maximumTorsoResidualAction)
                        / maximumTorsoResidualAction,
                    -0.999, 0.999)
            }
        }
        let asymmetryOffset = torsoOffset + torsoResidualKnotCount
        if armAsymmetryKnotCount > 0 {
            for knot in 1...armAsymmetryKnotCount {
                let progress = sourceProgress(
                    knot: knot, count: armAsymmetryKnotCount)
                for action in 0..<4 {
                    shifted[asymmetryOffset + (knot - 1) * 4 + action] =
                        simd_clamp(
                            armAsymmetryAction(
                                parameters, action: action,
                                progress: progress,
                                armParameterCount: armParameterCount,
                                blendKnotCount: blendKnotCount,
                                legResidualKnotCount: legResidualKnotCount,
                                torsoResidualKnotCount: torsoResidualKnotCount,
                                asymmetryKnotCount: armAsymmetryKnotCount,
                                maximumAction: maximumArmAsymmetryAction)
                                / maximumArmAsymmetryAction,
                            -0.999, 0.999)
                }
            }
        }
        return shifted
    }

    static func legResidualAction(
        _ parameters: [Float], action: Int, progress: Float,
        armParameterCount: Int, blendKnotCount: Int,
        residualKnotCount: Int, maximumAction: Float
    ) -> Float {
        precondition((0..<10).contains(action))
        precondition(residualKnotCount > 0)
        precondition(maximumAction > 0)
        let offset = armParameterCount + blendKnotCount
        precondition(parameters.count
            >= offset + 10 * residualKnotCount)
        let scaled = simd_clamp(progress, 0, 1)
            * Float(residualKnotCount)
        let lower = min(Int(floor(scaled)), residualKnotCount - 1)
        let fraction = scaled - Float(lower)
        func knot(_ index: Int) -> Float {
            guard index > 0 else { return 0 }
            return simd_clamp(parameters[
                offset + (index - 1) * 10 + action], -1, 1)
                * maximumAction
        }
        return (1 - fraction) * knot(lower)
            + fraction * knot(min(lower + 1, residualKnotCount))
    }

    static func torsoResidualAction(
        _ parameters: [Float], progress: Float,
        armParameterCount: Int, blendKnotCount: Int,
        legResidualKnotCount: Int, torsoResidualKnotCount: Int,
        maximumAction: Float
    ) -> Float {
        precondition(torsoResidualKnotCount > 0 && maximumAction > 0)
        let offset = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount
        precondition(parameters.count >= offset + torsoResidualKnotCount)
        let scaled = simd_clamp(progress, 0, 1)
            * Float(torsoResidualKnotCount)
        let lower = min(
            Int(floor(scaled)), torsoResidualKnotCount - 1)
        let fraction = scaled - Float(lower)
        func knot(_ index: Int) -> Float {
            guard index > 0 else { return 0 }
            return simd_clamp(parameters[offset + index - 1], -1, 1)
                * maximumAction
        }
        return (1 - fraction) * knot(lower)
            + fraction * knot(min(lower + 1, torsoResidualKnotCount))
    }

    static func armAsymmetryAction(
        _ parameters: [Float], action: Int, progress: Float,
        armParameterCount: Int, blendKnotCount: Int,
        legResidualKnotCount: Int, torsoResidualKnotCount: Int,
        asymmetryKnotCount: Int, maximumAction: Float
    ) -> Float {
        precondition((0..<4).contains(action))
        precondition(asymmetryKnotCount > 0 && maximumAction > 0)
        let offset = armParameterCount + blendKnotCount
            + 10 * legResidualKnotCount + torsoResidualKnotCount
        precondition(parameters.count >= offset + 4 * asymmetryKnotCount)
        let scaled = simd_clamp(progress, 0, 1)
            * Float(asymmetryKnotCount)
        let lower = min(Int(floor(scaled)), asymmetryKnotCount - 1)
        let fraction = scaled - Float(lower)
        func knot(_ index: Int) -> Float {
            guard index > 0 else { return 0 }
            return simd_clamp(parameters[
                offset + (index - 1) * 4 + action], -1, 1)
                * maximumAction
        }
        return (1 - fraction) * knot(lower)
            + fraction * knot(min(lower + 1, asymmetryKnotCount))
    }

    /// Re-express a verified trajectory with a different number of leg
    /// residual knots without changing its piecewise-linear control signal.
    /// A coarse certified flow can therefore remain the immutable prefix of a
    /// richer continuation search instead of being silently reinterpreted.
    public static func resampledLegResidualTrajectory(
        _ parameters: [Float], armParameterCount: Int,
        blendKnotCount: Int, sourceResidualKnotCount: Int,
        targetResidualKnotCount: Int,
        sourceResidualMaximumAction: Float = 1,
        targetResidualMaximumAction: Float = 1,
        sourceTorsoResidualKnotCount: Int = 0,
        targetTorsoResidualKnotCount: Int = 0,
        sourceTorsoResidualMaximumAction: Float = 1,
        targetTorsoResidualMaximumAction: Float = 1,
        sourceArmAsymmetryKnotCount: Int = 0,
        targetArmAsymmetryKnotCount: Int = 0,
        sourceArmAsymmetryMaximumAction: Float = 1,
        targetArmAsymmetryMaximumAction: Float = 1
    ) -> [Float] {
        precondition(armParameterCount > 0
            && armParameterCount.isMultiple(of: 4))
        precondition(blendKnotCount >= 0)
        precondition(sourceResidualKnotCount >= 0)
        precondition(targetResidualKnotCount >= 0)
        precondition(sourceTorsoResidualKnotCount >= 0)
        precondition(targetTorsoResidualKnotCount >= 0)
        precondition(sourceArmAsymmetryKnotCount >= 0)
        precondition(targetArmAsymmetryKnotCount >= 0)
        precondition(sourceResidualMaximumAction.isFinite
            && sourceResidualMaximumAction > 0)
        precondition(targetResidualMaximumAction.isFinite
            && targetResidualMaximumAction > 0)
        precondition(sourceTorsoResidualMaximumAction.isFinite
            && sourceTorsoResidualMaximumAction > 0)
        precondition(targetTorsoResidualMaximumAction.isFinite
            && targetTorsoResidualMaximumAction > 0)
        precondition(sourceArmAsymmetryMaximumAction.isFinite
            && sourceArmAsymmetryMaximumAction > 0)
        precondition(targetArmAsymmetryMaximumAction.isFinite
            && targetArmAsymmetryMaximumAction > 0)
        precondition(parameters.count == armParameterCount + blendKnotCount
            + 10 * sourceResidualKnotCount
            + sourceTorsoResidualKnotCount
            + 4 * sourceArmAsymmetryKnotCount)
        var result = Array(parameters.prefix(
            armParameterCount + blendKnotCount))
        result.reserveCapacity(result.count + 10 * targetResidualKnotCount
            + targetTorsoResidualKnotCount
            + 4 * targetArmAsymmetryKnotCount)
        for knot in 0..<targetResidualKnotCount {
            let progress = Float(knot + 1)
                / Float(targetResidualKnotCount)
            for action in 0..<10 {
                if sourceResidualKnotCount == 0 {
                    result.append(0)
                } else {
                    result.append(legResidualAction(
                        parameters, action: action, progress: progress,
                        armParameterCount: armParameterCount,
                        blendKnotCount: blendKnotCount,
                        residualKnotCount: sourceResidualKnotCount,
                        maximumAction:
                            sourceResidualMaximumAction
                                / targetResidualMaximumAction))
                }
            }
        }
        for knot in 0..<targetTorsoResidualKnotCount {
            let progress = Float(knot + 1)
                / Float(targetTorsoResidualKnotCount)
            if sourceTorsoResidualKnotCount == 0 {
                result.append(0)
            } else {
                result.append(torsoResidualAction(
                    parameters, progress: progress,
                    armParameterCount: armParameterCount,
                    blendKnotCount: blendKnotCount,
                    legResidualKnotCount: sourceResidualKnotCount,
                    torsoResidualKnotCount:
                        sourceTorsoResidualKnotCount,
                    maximumAction:
                        sourceTorsoResidualMaximumAction
                            / targetTorsoResidualMaximumAction))
            }
        }
        for knot in 0..<targetArmAsymmetryKnotCount {
            let progress = Float(knot + 1)
                / Float(targetArmAsymmetryKnotCount)
            for action in 0..<4 {
                if sourceArmAsymmetryKnotCount == 0 {
                    result.append(0)
                } else {
                    result.append(armAsymmetryAction(
                        parameters, action: action, progress: progress,
                        armParameterCount: armParameterCount,
                        blendKnotCount: blendKnotCount,
                        legResidualKnotCount: sourceResidualKnotCount,
                        torsoResidualKnotCount:
                            sourceTorsoResidualKnotCount,
                        asymmetryKnotCount:
                            sourceArmAsymmetryKnotCount,
                        maximumAction:
                            sourceArmAsymmetryMaximumAction
                                / targetArmAsymmetryMaximumAction))
                }
            }
        }
        return result
    }

    /// Goal-set score used only to discover a new physically valid frontier
    /// state. The subsequent reconstruction still targets the exact selected
    /// simulator state and never receives its generating trajectory.
    private static func frontierEvaluation(
        state: State, flags: Flags, clearance: Float,
        carryDistance: Float, minimumCarryDistance: Float,
        destinationProgress: Float, minimumDestinationProgress: Float,
        objectiveDestinationProgress: Float,
        rootDestinationProgress: Float,
        minimumRootDestinationProgress: Float,
        objectiveRootDestinationProgress: Float,
        loadedTouchdowns: Int, minimumTouchdowns: Int,
        loadedAlternatingSteps: Int, minimumAlternatingSteps: Int,
        maximumSwingFootLift: Float, minimumSwingFootLift: Float,
        objectiveSwingFootLift: Float,
        maximumFootAirTime: Float, minimumFootAirTime: Float,
        objectiveFootAirTime: Float,
        maximumFootUnloading: Float, minimumFootUnloading: Float,
        objectiveFootUnloading: Float,
        minimumClearance: Float, objectiveClearance: Float,
        minimumGraspQuality: Float, objectiveGraspQuality: Float,
        feasibilityDwellSteps: Int, requiredDwellSteps: Int,
        feasibilityWindowPenalty: Float,
        stablePathViolationSteps: Int, trajectorySteps: Int,
        requireStableCarryPath: Bool
    ) -> PhysicalFlowEndpointEvaluation {
        let rootUp = state.humanoid.root.rotation
            .act(F3(0, 0, 1)).z
        let boxUp = state.manipulation.object.rotation
            .act(F3(0, 0, 1)).z
        var errors: [Float] = [
            max(minimumCarryDistance - carryDistance, 0) / 0.02,
            max(objectiveClearance - clearance, 0)
                / max(objectiveClearance, 0.02),
            max(objectiveGraspQuality - flags.graspQuality, 0)
                / max(objectiveGraspQuality, 0.25),
            length(state.manipulation.object.linearVelocity) / 0.20,
            length(state.manipulation.object.angularVelocity) / 0.60,
            length(state.humanoid.root.linearVelocity) / 0.40,
            length(state.humanoid.root.angularVelocity) / 0.80,
            max(0.9 - rootUp, 0) / 0.10,
            max(0.9 - boxUp, 0) / 0.10,
            // These are feasibility barriers, not tradeable rewards. A box
            // flying toward the goal after contact loss must never outrank a
            // shorter but valid carry state.
            hardMinimumPenalty(
                value: flags.graspFrictionSupportFraction,
                minimum: 1, scale: 0.25),
            flags.unsupported ? 0 : 10,
            flags.physicallyLifted ? 0 : 10,
            rootUp > 0.9 ? 0 : 10,
            boxUp > 0.9 ? 0 : 10,
            clearance >= minimumClearance ? 0 : 10,
            // Preserve a hard >1 feasibility barrier while still ranking
            // sub-threshold grasps by how much opposing-face error remains.
            // A flat Boolean penalty made every slipping contact equally bad,
            // so CEM optimized secondary velocity terms instead of closing the
            // final few centimetres of hand-to-face distance.
            hardMinimumPenalty(
                value: flags.graspQuality,
                minimum: minimumGraspQuality,
                scale: 0.05),
            hardDwellPenalty(
                achieved: feasibilityDwellSteps,
                required: requiredDwellSteps),
            feasibilityWindowPenalty,
            hardViolationPenalty(
                violations: requireStableCarryPath
                    ? stablePathViolationSteps : 0,
                totalSteps: trajectorySteps),
            flags.failed ? 20 : 0,
        ]
        if objectiveDestinationProgress > 0 {
            errors.append(max(
                objectiveDestinationProgress - destinationProgress, 0)
                / max(objectiveDestinationProgress, 0.05))
        }
        if minimumDestinationProgress > 0 {
            errors.append(hardMinimumPenalty(
                value: destinationProgress,
                minimum: minimumDestinationProgress,
                scale: 0.01))
        }
        if objectiveRootDestinationProgress > 0 {
            errors.append(max(
                objectiveRootDestinationProgress - rootDestinationProgress,
                0) / max(objectiveRootDestinationProgress, 0.05))
        }
        if minimumRootDestinationProgress > 0 {
            errors.append(hardMinimumPenalty(
                value: rootDestinationProgress,
                minimum: minimumRootDestinationProgress,
                scale: 0.01))
        }
        errors.append(hardIntegerMinimumPenalty(
            value: loadedTouchdowns,
            minimum: minimumTouchdowns))
        errors.append(hardIntegerMinimumPenalty(
            value: loadedAlternatingSteps,
            minimum: minimumAlternatingSteps))
        if minimumSwingFootLift > 0 {
            errors.append(hardMinimumPenalty(
                value: maximumSwingFootLift,
                minimum: minimumSwingFootLift,
                scale: 0.015))
        }
        if objectiveSwingFootLift > 0 {
            errors.append(max(
                objectiveSwingFootLift - maximumSwingFootLift, 0)
                / 0.01)
        }
        if objectiveFootAirTime > 0 {
            errors.append(max(
                objectiveFootAirTime - maximumFootAirTime, 0)
                / 0.02)
        }
        if minimumFootAirTime > 0 {
            errors.append(hardMinimumPenalty(
                value: maximumFootAirTime,
                minimum: minimumFootAirTime,
                scale: 0.02))
        }
        if objectiveFootUnloading > 0 {
            errors.append(max(
                objectiveFootUnloading - maximumFootUnloading, 0)
                / 0.10)
        }
        if minimumFootUnloading > 0 {
            errors.append(hardMinimumPenalty(
                value: maximumFootUnloading,
                minimum: minimumFootUnloading,
                scale: 0.10))
        }
        return PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: errors)
    }

    private static func metrics(
        from state: State, to target: State, flags: Flags,
        carryDistance: Float, targetCarryDistance: Float,
        minimumCarryDistance: Float, clearance: Float,
        minimumClearance: Float, graspQuality: Float,
        minimumGraspQuality: Float, feasibilityDwellSteps: Int,
        requiredDwellSteps: Int, feasibilityWindowPenalty: Float = 0,
        stablePathViolationSteps: Int,
        stablePathRequired: Bool, trajectorySteps: Int,
        terminalFootUnloadingFraction: Float? = nil,
        minimumTerminalFootUnloadingFraction: Float? = nil
    ) -> HumanoidBoxPhysicalFlowMetrics {
        let endpoint = stateEndpointEvaluation(from: state, to: target)
        var errors = endpoint.normalizedErrors
        let carryDistanceError = abs(carryDistance - targetCarryDistance)
        errors.append(carryDistanceError / 0.05)
        errors.append(hardMinimumPenalty(
            value: flags.graspFrictionSupportFraction,
            minimum: 1, scale: 0.25))
        errors.append(flags.unsupported ? 0 : 2)
        errors.append(flags.physicallyLifted ? 0 : 2)
        errors.append(flags.robotUpright ? 0 : 2)
        errors.append(flags.boxUpright ? 0 : 2)
        errors.append(flags.failed ? 4 : 0)
        let minimumCarryAchieved = carryDistance >= minimumCarryDistance
        // Keep a smooth search signal up to the exact task boundary. The
        // feasibility barrier makes every passing candidate outrank a near
        // miss instead of letting lower velocity errors hide a hard failure.
        // The endpoint classifier below independently retains the exact
        // unrelaxed Boolean boundary.
        errors.append(hardMinimumPenalty(
            value: carryDistance, minimum: minimumCarryDistance,
            scale: 0.01))
        let minimumClearanceAchieved = clearance >= minimumClearance
        errors.append(hardMinimumPenalty(
            value: clearance, minimum: minimumClearance,
            scale: 0.01))
        let minimumGraspQualityAchieved =
            graspQuality >= minimumGraspQuality
        errors.append(hardMinimumPenalty(
            value: graspQuality, minimum: minimumGraspQuality,
            scale: 0.05))
        let minimumTerminalFootUnloadingAchieved =
            minimumTerminalFootUnloadingFraction.map { minimum in
                terminalFootUnloadingFraction.map { $0 >= minimum } ?? false
            }
        if let minimum = minimumTerminalFootUnloadingFraction {
            errors.append(hardMinimumPenalty(
                value: terminalFootUnloadingFraction ?? 0,
                minimum: minimum, scale: 0.10))
        }
        let minimumDwellAchieved = feasibilityDwellSteps
            >= requiredDwellSteps
        errors.append(hardDwellPenalty(
            achieved: feasibilityDwellSteps, required: requiredDwellSteps))
        errors.append(feasibilityWindowPenalty)
        let stablePathAchieved = !stablePathRequired
            || stablePathViolationSteps == 0
        errors.append(hardViolationPenalty(
            violations: stablePathRequired ? stablePathViolationSteps : 0,
            totalSteps: trajectorySteps))
        let constrained = PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: errors)
        let components = stateErrors(from: state, to: target)
        return HumanoidBoxPhysicalFlowMetrics(
            loss: constrained.bottleneckLoss,
            maximumNormalizedError: constrained.maximumNormalizedError,
            rootPositionErrorMeters: components[0],
            rootRotationErrorRadians: components[1],
            rootLinearVelocityErrorMPS: components[2],
            rootAngularVelocityErrorRadPS: components[3],
            jointAngleRMSErrorRadians: components[4],
            jointVelocityRMSErrorRadPS: components[5],
            maximumFootPositionErrorMeters: components[6],
            maximumFootVelocityErrorMPS: components[7],
            boxPositionErrorMeters: components[8],
            boxRotationErrorRadians: components[9],
            boxLinearVelocityErrorMPS: components[10],
            boxAngularVelocityErrorRadPS: components[11],
            maximumHandPositionErrorMeters: components[12],
            maximumHandVelocityErrorMPS: components[13],
            carryDistanceErrorMeters: carryDistanceError,
            clearanceMeters: clearance,
            graspQuality: graspQuality,
            bilateralHandContact: flags.bilateral,
            unsupported: flags.unsupported,
            physicallyLifted: flags.physicallyLifted,
            robotUpright: flags.robotUpright,
            boxUpright: flags.boxUpright,
            failed: flags.failed,
            minimumCarryDistanceAchieved: minimumCarryAchieved,
            minimumClearanceAchieved: minimumClearanceAchieved,
            minimumGraspQualityAchieved:
                minimumGraspQualityAchieved,
            feasibilityDwellSteps: feasibilityDwellSteps,
            minimumDwellAchieved: minimumDwellAchieved,
            feasibilityWindowMaximumNormalizedError:
                feasibilityWindowPenalty,
            stablePathViolationSteps: stablePathViolationSteps,
            stablePathAchieved: stablePathAchieved,
            terminalFootUnloadingFraction:
                terminalFootUnloadingFraction,
            minimumTerminalFootUnloadingAchieved:
                minimumTerminalFootUnloadingAchieved)
    }

    static func hardMinimumPenalty(
        value: Float, minimum: Float, scale: Float
    ) -> Float {
        precondition(minimum.isFinite && scale.isFinite && scale > 0)
        guard value < minimum else { return 0 }
        return 4 + (minimum - value) / scale
    }

    static func hardIntegerMinimumPenalty(
        value: Int, minimum: Int
    ) -> Float {
        precondition(value >= 0 && minimum >= 0)
        guard value < minimum else { return 0 }
        return 4 + Float(minimum - value)
    }

    static func minimumDestinationProgressPassed(
        _ value: Float, minimum: Float
    ) -> Bool {
        minimum <= 0 || value >= minimum
    }

    static func scheduledProgressMinimum(
        finalMinimum: Float, absoluteStep: Int, executionSteps: Int
    ) -> Float {
        precondition(finalMinimum.isFinite && finalMinimum >= 0)
        precondition(absoluteStep > 0 && executionSteps > 0)
        return finalMinimum * min(
            Float(absoluteStep) / Float(executionSteps), 1)
    }

    static func scheduledAbsoluteMinimum(
        initial: Float, finalMinimum: Float,
        absoluteStep: Int, executionSteps: Int
    ) -> Float {
        precondition(initial.isFinite)
        precondition(finalMinimum.isFinite && finalMinimum >= 0)
        precondition(absoluteStep > 0 && executionSteps > 0)
        let progress = min(
            Float(absoluteStep) / Float(executionSteps), 1)
        return initial + (finalMinimum - initial) * progress
    }

    static func scheduledIntegerMinimum(
        finalMinimum: Int, absoluteStep: Int, executionSteps: Int
    ) -> Int {
        precondition(finalMinimum >= 0)
        precondition(absoluteStep > 0 && executionSteps > 0)
        return Int(floor(Float(finalMinimum) * min(
            Float(absoluteStep) / Float(executionSteps), 1)))
    }

    static func recoveryControlFeasible(
        physicalStable: Bool,
        carryDistance: Float, minimumCarryDistance: Float,
        destinationProgress: Float, minimumDestinationProgress: Float,
        clearance: Float, minimumClearance: Float,
        graspQuality: Float, minimumGraspQuality: Float
    ) -> Bool {
        recoveryRetentionFeasible(
            physicalStable: physicalStable,
            carryDistance: carryDistance,
            minimumCarryDistance: minimumCarryDistance,
            clearance: clearance,
            minimumClearance: minimumClearance,
            graspQuality: graspQuality,
            minimumGraspQuality: minimumGraspQuality)
            && minimumDestinationProgressPassed(
                destinationProgress, minimum: minimumDestinationProgress)
    }

    static func recoveryRetentionFeasible(
        physicalStable: Bool,
        carryDistance: Float, minimumCarryDistance: Float,
        clearance: Float, minimumClearance: Float,
        graspQuality: Float, minimumGraspQuality: Float
    ) -> Bool {
        physicalStable
            && carryDistance >= minimumCarryDistance
            && clearance >= minimumClearance
            && graspQuality >= minimumGraspQuality
    }

    static func terminalRecoveryViable(
        clearance: Float, minimumClearance: Float?,
        boxVerticalVelocity: Float, maximumDownwardVelocity: Float?,
        footUnloadingFraction: Float = 0,
        minimumFootUnloading: Float? = nil
    ) -> Bool {
        (minimumClearance.map { clearance >= $0 } ?? true)
            && (maximumDownwardVelocity.map {
                boxVerticalVelocity >= -$0
            } ?? true)
            && (minimumFootUnloading.map {
                footUnloadingFraction >= $0
            } ?? true)
    }

    static func pathDownwardVelocityFeasible(
        boxVerticalVelocity: Float, maximumDownwardVelocity: Float?
    ) -> Bool {
        maximumDownwardVelocity.map {
            boxVerticalVelocity >= -$0
        } ?? true
    }

    static func pathDownwardVelocityPenalty(
        boxVerticalVelocity: Float, maximumDownwardVelocity: Float?
    ) -> Float {
        maximumDownwardVelocity.map {
            hardMinimumPenalty(
                value: boxVerticalVelocity, minimum: -$0, scale: 0.05)
        } ?? 0
    }

    /// Damped least-squares task-space correction for a small serial chain.
    /// Jacobian columns are world-space hand displacement per joint radian.
    static func dampedLeastSquaresJointDelta(
        jacobian: [F3], taskDelta: F3, damping: Float
    ) -> [Float] {
        precondition(!jacobian.isEmpty)
        precondition(damping.isFinite && damping > 0)
        let d2 = damping * damping
        var a00 = d2
        var a01: Float = 0
        var a02: Float = 0
        var a11 = d2
        var a12: Float = 0
        var a22 = d2
        for column in jacobian {
            a00 += column.x * column.x
            a01 += column.x * column.y
            a02 += column.x * column.z
            a11 += column.y * column.y
            a12 += column.y * column.z
            a22 += column.z * column.z
        }
        let c00 = a11 * a22 - a12 * a12
        let c01 = a02 * a12 - a01 * a22
        let c02 = a01 * a12 - a02 * a11
        let c11 = a00 * a22 - a02 * a02
        let c12 = a01 * a02 - a00 * a12
        let c22 = a00 * a11 - a01 * a01
        let determinant = a00 * c00 + a01 * c01 + a02 * c02
        guard determinant.isFinite, abs(determinant) > 1e-12 else {
            return [Float](repeating: 0, count: jacobian.count)
        }
        let inverseDeterminant = 1 / determinant
        let solved = F3(
            (c00 * taskDelta.x + c01 * taskDelta.y
                + c02 * taskDelta.z) * inverseDeterminant,
            (c01 * taskDelta.x + c11 * taskDelta.y
                + c12 * taskDelta.z) * inverseDeterminant,
            (c02 * taskDelta.x + c12 * taskDelta.y
                + c22 * taskDelta.z) * inverseDeterminant)
        return jacobian.map { dot($0, solved) }
    }

    static func isUnsupported(
        pedestalContact: Float, destinationContact: Float,
        groundContact: Float
    ) -> Bool {
        pedestalContact < 0.5 && destinationContact < 0.5
            && groundContact < 0.5
    }

    static func hardDwellPenalty(achieved: Int, required: Int) -> Float {
        precondition(achieved >= 0 && required > 0)
        guard achieved < required else { return 0 }
        // Dwell is a dependent constraint: it can only accumulate after the
        // instantaneous contact, carry, clearance, and posture prerequisites
        // pass. Rank those prerequisite boundaries first (their penalties are
        // >= 4), then expose a normalized > 1 dwell deficit. The separate
        // Boolean endpoint gate still requires the exact full dwell, so this
        // improves the optimizer's ordering without relaxing certification.
        return 1 + Float(required - achieved) / Float(required)
    }

    static func feasibilityWindowPenalty(
        _ instantaneousPenalties: [Float], required: Int
    ) -> Float {
        precondition(required > 0
            && instantaneousPenalties.count <= required
            && instantaneousPenalties.allSatisfy {
                $0.isFinite && $0 >= 0
            })
        guard instantaneousPenalties.count == required else {
            return 4 + Float(required - instantaneousPenalties.count)
                / Float(required)
        }
        return instantaneousPenalties.max() ?? 0
    }

    static func appendFeasibilityPenalty(
        _ penalty: Float, to window: inout [Float], required: Int
    ) {
        precondition(penalty.isFinite && penalty >= 0 && required > 0)
        window.append(penalty)
        if window.count > required {
            window.removeFirst(window.count - required)
        }
    }

    static func hardViolationPenalty(
        violations: Int, totalSteps: Int
    ) -> Float {
        precondition(violations >= 0 && totalSteps > 0)
        guard violations > 0 else { return 0 }
        return 4 + Float(violations) / Float(totalSteps)
    }

    private static func stateEndpointEvaluation(
        from state: State, to target: State
    ) -> PhysicalFlowEndpointEvaluation {
        let e = stateErrors(from: state, to: target)
        let thresholds: [Float] = [
            0.05, 0.12, 0.15, 0.30,
            0.12, 0.50, 0.06, 0.25,
            0.04, 0.12, 0.12, 0.40,
            0.05, 0.25,
        ]
        return PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: zip(e, thresholds).map { $0 / $1 })
    }

    private static func stateErrors(
        from state: State, to target: State
    ) -> [Float] {
        func rmse(_ lhs: [Float], _ rhs: [Float]) -> Float {
            sqrt(zip(lhs, rhs).reduce(0) {
                let d = $1.0 - $1.1
                return $0 + d * d
            } / Float(lhs.count))
        }
        func maximum(_ values: Float...) -> Float { values.max() ?? 0 }
        let h = state.humanoid, t = target.humanoid
        let m = state.manipulation, tm = target.manipulation
        return [
            length(h.root.position - t.root.position),
            rotationError(h.root.rotation, t.root.rotation),
            length(h.root.linearVelocity - t.root.linearVelocity),
            length(h.root.angularVelocity - t.root.angularVelocity),
            rmse(h.jointAngles, t.jointAngles),
            rmse(h.jointVelocities, t.jointVelocities),
            maximum(
                length(h.leftFoot.position - t.leftFoot.position),
                length(h.rightFoot.position - t.rightFoot.position)),
            maximum(
                length(h.leftFoot.linearVelocity
                    - t.leftFoot.linearVelocity),
                length(h.rightFoot.linearVelocity
                    - t.rightFoot.linearVelocity)),
            length(m.object.position - tm.object.position),
            rotationError(m.object.rotation, tm.object.rotation),
            length(m.object.linearVelocity - tm.object.linearVelocity),
            length(m.object.angularVelocity - tm.object.angularVelocity),
            maximum(
                length(m.leftHand.position - tm.leftHand.position),
                length(m.rightHand.position - tm.rightHand.position)),
            maximum(
                length(m.leftHand.linearVelocity
                    - tm.leftHand.linearVelocity),
                length(m.rightHand.linearVelocity
                    - tm.rightHand.linearVelocity)),
        ]
    }

    private static func rotationError(
        _ lhs: simd_quatf, _ rhs: simd_quatf
    ) -> Float {
        let cosine = simd_clamp(abs(simd_dot(lhs.vector, rhs.vector)), 0, 1)
        return 2 * acos(cosine)
    }

    private static func cholesky(_ covariance: [[Float]]) -> [[Float]] {
        let count = covariance.count
        var lower = [[Float]](
            repeating: [Float](repeating: 0, count: count), count: count)
        for row in 0..<count {
            for column in 0...row {
                var value = covariance[row][column]
                for k in 0..<column {
                    value -= lower[row][k] * lower[column][k]
                }
                if row == column {
                    lower[row][column] = sqrt(max(value, 1e-8))
                } else {
                    lower[row][column] = value
                        / max(lower[column][column], 1e-8)
                }
            }
        }
        return lower
    }
}
