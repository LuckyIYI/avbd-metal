import AVBDCore
import AVBDLearn
import Metal
import Foundation
import simd

// avbd — headless CLI for the Metal AVBD solver.
//
// Usage:
//   avbd run <demo> [--frames N] [--iterations N] [--cpu] [--json] [--scale N]
//            [--dt T] [--watch BODY] [--stats-every N]
//   avbd bench <demo> [--frames N] [--scale N] [--iterations N]
//   avbd list
//   avbd parity <demo> [--frames N]

struct Options {
    var frames = 300
    var iterations: Int? = nil
    var scale = 1
    var dt: Float? = nil
    var useCPU = false
    var json = false
    var watch: Int? = nil
    var res: Int? = nil
    var statsEvery = 60
    var envs = 256
    var batch = 1024
    var latent = 128
    var lr: Float = 3e-4
    var optimizerEpsilon: Float? = nil
    var gamma: Float = 0.99
    var gaeLambda: Float = 0.95
    var rewardScale: Float = 1
    var lambda: Float = 0.5
    var episodes = 10
    var updates = 3000
    var horizon = 24
    var epochs = 5
    var hidden = 512
    var hiddenLayers: [Int]? = nil
    var actionDistribution: PPOActionDistribution? = nil
    var seed: UInt64 = 1
    var algorithm = "ppo"
    var actionStd: Float = 1.0
    var minimumActionStd: Float? = nil
    var maximumActionStd: Float? = nil
    var finalActionStd: Float? = nil
    var actionStdAnnealStartUpdate: Int? = nil
    var actionStdAnnealEndUpdate: Int? = nil
    var initializeFrom: String? = nil
    var policyExpertFrom: String? = nil
    var policyExpertBranchFrom: String? = nil
    var standExpertFrom: String? = nil
    var locomotionCheckpoint: String? = nil
    var initializationNormalizerPriorCount: Double? = nil
    var symmetryAugmentation = true
    var symmetryMirrorLossCoefficient: Float? = 0.01
    var updateObservationNormalizer = true
    var normalizeObservations = true
    var entropy: Float = 0.01
    var targetKL: Float = 0.01
    var klSchedule: PPOKLSchedule? = nil
    var policyClip: Float = 0.2
    var valueClip: Float = 0.2
    var clipValueLoss = true
    var valueCoefficient: Float = 1
    var maxGradientNorm: Float = 1
    var successImitationCoefficient: Float = 0
    var successReplayCapacity = 0
    var successReplayBatchSize = 0
    var successImitationHistorySteps: Int?
    var referencePolicyCoefficient: Float = 0
    var checkpointInterval = 50
    var activation: PPOActivation? = nil
    var orthogonalInitialization = false
    var actorOutputGain: Float? = nil
    var preset: String? = nil
    var resume = false
    var runName = "ppo"
    var checkpoint: String? = nil
    var output: String? = nil
    var allowTaskTransfer = false
    var expectedCheckpointFingerprint: String? = nil
    var sourceCommit: String? = nil
    var inferenceBatchSize: Int? = nil
    var evaluationSeeds: [UInt64]? = nil
    var evaluationEnvironments: Int? = nil
    var parentCheckpoint: String? = nil
    var qualificationDirectory: String? = nil
    /// Task-owned numeric configuration. Keeping these values out of the PPO
    /// parser lets new scenes expose curricula and control settings without
    /// adding task-specific branches to this executable.
    var taskOptions: [String: Float] = [:]
    var gaitSwingSteps = 4
    var gaitSwingHeight: Float = 0.016
    var gaitPlacementHorizon: Float = 0.35
    var gaitMaximumPlacement: Float = 0.045
}

/// Reads the action trajectory from either a probe or physical-flow artifact
/// without requiring every diagnostic field in that artifact to match the
/// current report schema. Experiment results are durable inputs: adding a new
/// metric must not invalidate a previously verified physical trajectory.
func humanoidBoxTrajectory(from data: Data) -> [Float]? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any] else {
        return nil
    }
    for key in ["bestTrajectory", "bestArmTarget"] {
        guard let values = dictionary[key] as? [NSNumber] else { continue }
        let trajectory = values.map(\.floatValue)
        guard !trajectory.isEmpty,
              trajectory.allSatisfy(\.isFinite) else { continue }
        return trajectory
    }
    return nil
}

private struct HumanoidBoxFlowQualification {
    let finitePrefixPassed: Bool
    let recoveryPathSafe: Bool
    let reusableFrontierPassed: Bool

    init(_ dictionary: [String: Any]) {
        finitePrefixPassed = (dictionary[
            "targetFinitePrefixGatePassed"] as? Bool)
            ?? (dictionary["targetPlanningGatePassed"] as? Bool)
            ?? ((dictionary["targetGatePassed"] as? Bool) == true
                && ((dictionary["targetCloneSuccessFraction"]
                    as? NSNumber)?.floatValue ?? 0) >= 0.8
                && ((dictionary["targetReplayMaximumNormalizedError"]
                    as? NSNumber)?.floatValue ?? .infinity) < 0.02)
        recoveryPathSafe = (dictionary[
            "targetPredictedRecoveryPathSafe"] as? Bool)
            // Open-loop legacy reports had no continuation boundary and no
            // recovery field. Preserve that historical interpretation only
            // when the artifact also has no receding-horizon schema.
            ?? (dictionary["recedingHorizonSteps"] == nil)
        reusableFrontierPassed = (dictionary[
            "targetReusableFrontierGatePassed"] as? Bool)
            ?? (finitePrefixPassed && recoveryPathSafe)
    }
}

private func humanoidBoxDeclaredTrajectoryElementCount(
    _ dictionary: [String: Any]
) -> Int? {
    for key in [
        "targetGeneratingTrajectory", "bestTrajectory", "bestArmTarget",
    ] {
        if let values = dictionary[key] as? [NSNumber], !values.isEmpty {
            return values.count
        }
    }
    return nil
}

/// Reads an optional committed-action trace without silently dropping malformed
/// samples. `present == true && actions == nil` means the field existed but was
/// not an exact 1...N sequence of full-body controls.
private func humanoidBoxCommittedActions(
    _ value: Any?, expectedCount: Int
) -> (present: Bool, actions: [[Float]]?) {
    guard let value else { return (false, nil) }
    guard expectedCount > 0,
          let trace = value as? [[String: Any]],
          trace.count == expectedCount
            || trace.count == expectedCount + 1 else {
        return (true, nil)
    }
    var samples = [(step: Int, action: [Float])]()
    samples.reserveCapacity(expectedCount)
    var sawBoundarySample = false
    for sample in trace {
        guard let step = (sample["step"] as? NSNumber)?.intValue else {
            return (true, nil)
        }
        if step == 0 {
            guard !sawBoundarySample,
                  sample["appliedNormalizedActions"] == nil else {
                return (true, nil)
            }
            sawBoundarySample = true
            continue
        }
        guard let encodedAction = sample[
            "appliedNormalizedActions"] as? [NSNumber] else {
            return (true, nil)
        }
        let action = encodedAction.map(\.floatValue)
        guard action.count == HumanoidBoxCarryTask.actionDimension,
              action.allSatisfy(\.isFinite) else {
            return (true, nil)
        }
        samples.append((step, action))
    }
    samples.sort { $0.step < $1.step }
    guard samples.count == expectedCount,
          samples.enumerated().allSatisfy({ index, sample in
        sample.step == index + 1
    }) else {
        return (true, nil)
    }
    return (true, samples.map(\.action))
}

/// Resolves a physical-flow artifact into the complete simulator-executed
/// lineage needed to reproduce its terminal state. Older single-stage flow
/// reports are accepted and become a one-element lineage.
func humanoidBoxFlowSourceStages(
    from data: Data, useTargetGeneratingTrajectory: Bool = false,
    includeOwnStage: Bool = true
) -> [HumanoidBoxPhysicalFlowStage]? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any] else {
        return nil
    }
    if includeOwnStage && useTargetGeneratingTrajectory {
        let qualification = HumanoidBoxFlowQualification(dictionary)
        guard qualification.reusableFrontierPassed,
              ((dictionary["targetGenerationSteps"]
                as? NSNumber)?.intValue ?? 0) > 0 else { return nil }
    }
    if includeOwnStage && !useTargetGeneratingTrajectory,
       let passed = dictionary["goGatePassed"] as? Bool, !passed {
        return nil
    }
    var stages = [HumanoidBoxPhysicalFlowStage]()
    let flattenedSourceActions = (dictionary[
        "sourceAppliedActions"] as? [[NSNumber]])?.map {
            $0.map(\.floatValue)
        }
    var flattenedSourceActionOffset = 0
    if flattenedSourceActions != nil,
       dictionary["sourceStages"] as? [[String: Any]] == nil {
        return nil
    }
    if let encodedStages = dictionary["sourceStages"]
            as? [[String: Any]] {
        let expectedSourceActionCount = encodedStages.reduce(0) {
            $0 + (($1["controlSteps"] as? NSNumber)?.intValue ?? 0)
        }
        guard flattenedSourceActions.map({ actions in
            actions.count == expectedSourceActionCount
                && actions.allSatisfy {
                    $0.count == HumanoidBoxCarryTask.actionDimension
                        && $0.allSatisfy(\.isFinite)
                }
        }) ?? true else { return nil }
        for encoded in encodedStages {
            guard let steps = encoded["controlSteps"] as? NSNumber,
                  steps.intValue > 0 else { return nil }
            let minimumCarry = (encoded["minimumCarryDistanceMeters"]
                as? NSNumber)?.floatValue
            let minimumDestinationProgress = (encoded[
                "minimumDestinationProgressMeters"]
                as? NSNumber)?.floatValue
            let minimumRootDestinationProgress = (encoded[
                "minimumRootDestinationProgressMeters"]
                as? NSNumber)?.floatValue
            let minimumAlternatingSteps = (encoded[
                "minimumAlternatingSteps"]
                as? NSNumber)?.intValue
            let minimumSwingFootLift = (encoded[
                "minimumSwingFootLiftMeters"] as? NSNumber)?.floatValue
            let minimumTouchdowns = (encoded["minimumTouchdowns"]
                as? NSNumber)?.intValue
            let minimumFootAirTime = (encoded[
                "minimumFootAirTimeSeconds"] as? NSNumber)?.floatValue
            let minimumFootUnloading = (encoded[
                "minimumFootUnloadingFraction"] as? NSNumber)?.floatValue
            let minimumTerminalFootUnloading = (encoded[
                "minimumTerminalFootUnloadingFraction"]
                as? NSNumber)?.floatValue
            let minimumClearance = (encoded["minimumClearanceMeters"]
                as? NSNumber)?.floatValue
            let maximumPathDownwardBoxVelocity = (encoded[
                "maximumPathDownwardBoxVelocityMPS"]
                as? NSNumber)?.floatValue
            let minimumGraspQuality = (encoded["minimumGraspQuality"]
                as? NSNumber)?.floatValue
            let graspAnchorFeedbackBlend = (encoded[
                "graspAnchorFeedbackBlend"] as? NSNumber)?.floatValue
            let graspAnchorFeedbackVelocityHorizon = (encoded[
                "graspAnchorFeedbackVelocityHorizonSeconds"]
                as? NSNumber)?.floatValue
            let graspAnchorFeedbackMaximumActionCorrection = (encoded[
                "graspAnchorFeedbackMaximumActionCorrection"]
                as? NSNumber)?.floatValue
            let graspAnchorFeedbackInwardPreload = (encoded[
                "graspAnchorFeedbackInwardPreloadMeters"]
                as? NSNumber)?.floatValue
            let leftGraspAnchor = (encoded[
                "leftGraspAnchorBoxLocalMeters"] as? [NSNumber])?
                .map(\.floatValue)
            let rightGraspAnchor = (encoded[
                "rightGraspAnchorBoxLocalMeters"] as? [NSNumber])?
                .map(\.floatValue)
            let graspAnchorBoxHeight = (encoded[
                "graspAnchorBoxHeightMeters"] as? NSNumber)?.floatValue
            let certificationDwell = (encoded["certificationDwellSteps"]
                as? NSNumber)?.intValue
            let policyOnly = encoded["policyOnly"] as? Bool == true
            let canonicalizeReplicasBeforeExecution = encoded[
                "canonicalizeReplicasBeforeExecution"] as? Bool == true
            let continueFromPreviousTrajectoryTerminal = encoded[
                "continueFromPreviousTrajectoryTerminal"] as? Bool == true
            let forwardOnlyBaseCommand = encoded[
                "forwardOnlyBaseCommand"] as? Bool == true
            let holonomicBaseCommand = encoded[
                "holonomicBaseCommand"] as? Bool == true
            let locomotionCheckpointDirectory = encoded[
                "locomotionCheckpointDirectory"] as? String
            let locomotionCommandSpeed = (encoded[
                "locomotionCommandSpeed"] as? NSNumber)?.floatValue
            let encodedAppliedNormalizedActions = (encoded[
                "appliedNormalizedActions"] as? [[NSNumber]])?.map {
                    $0.map(\.floatValue)
                }
            guard encodedAppliedNormalizedActions.map({ actions in
                actions.count == steps.intValue
                    && actions.allSatisfy {
                        $0.count == HumanoidBoxCarryTask.actionDimension
                            && $0.allSatisfy(\.isFinite)
                    }
            }) ?? true else { return nil }
            let inheritedAppliedNormalizedActions:
                [[Float]]?
            let flattenedSourceActionEnd =
                flattenedSourceActionOffset + steps.intValue
            if let flattenedSourceActions,
               flattenedSourceActionEnd
                    <= flattenedSourceActions.count {
                inheritedAppliedNormalizedActions = Array(
                    flattenedSourceActions[
                        flattenedSourceActionOffset..<flattenedSourceActionEnd])
            } else {
                inheritedAppliedNormalizedActions = nil
            }
            flattenedSourceActionOffset += steps.intValue
            let appliedNormalizedActions =
                encodedAppliedNormalizedActions
                    ?? inheritedAppliedNormalizedActions
            if let encodedAppliedNormalizedActions,
               let inheritedAppliedNormalizedActions,
               encodedAppliedNormalizedActions
                    != inheritedAppliedNormalizedActions {
                return nil
            }
            if policyOnly {
                stages.append(.init(
                    trajectory: [], controlSteps: steps.intValue,
                    policyOnly: true,
                    appliedNormalizedActions:
                        appliedNormalizedActions,
                    canonicalizeReplicasBeforeExecution:
                        canonicalizeReplicasBeforeExecution,
                    continueFromPreviousTrajectoryTerminal:
                        continueFromPreviousTrajectoryTerminal,
                    graspAnchorFeedbackBlend:
                        graspAnchorFeedbackBlend,
                    graspAnchorFeedbackVelocityHorizonSeconds:
                        graspAnchorFeedbackVelocityHorizon,
                    graspAnchorFeedbackMaximumActionCorrection:
                        graspAnchorFeedbackMaximumActionCorrection,
                    graspAnchorFeedbackInwardPreloadMeters:
                        graspAnchorFeedbackInwardPreload,
                    leftGraspAnchorBoxLocalMeters: leftGraspAnchor,
                    rightGraspAnchorBoxLocalMeters: rightGraspAnchor,
                    graspAnchorBoxHeightMeters: graspAnchorBoxHeight,
                    minimumCarryDistanceMeters: minimumCarry,
                    minimumDestinationProgressMeters:
                        minimumDestinationProgress,
                    minimumRootDestinationProgressMeters:
                        minimumRootDestinationProgress,
                    minimumTouchdowns:
                        minimumTouchdowns,
                    minimumAlternatingSteps:
                        minimumAlternatingSteps,
                    minimumSwingFootLiftMeters:
                        minimumSwingFootLift,
                    minimumFootAirTimeSeconds:
                        minimumFootAirTime,
                    minimumFootUnloadingFraction:
                        minimumFootUnloading,
                    minimumTerminalFootUnloadingFraction:
                        minimumTerminalFootUnloading,
                    minimumClearanceMeters: minimumClearance,
                    maximumPathDownwardBoxVelocityMPS:
                        maximumPathDownwardBoxVelocity,
                    minimumGraspQuality: minimumGraspQuality,
                    certificationDwellSteps: certificationDwell))
                continue
            }
            if let encodedSequence = encoded[
                    "trajectorySequence"] as? [[NSNumber]] {
                let denominator = (encoded[
                    "trajectorySequenceStepDenominator"]
                        as? NSNumber)?.intValue
                let sequence = encodedSequence.map { $0.map(\.floatValue) }
                let phaseSteps = (encoded[
                    "trajectorySequencePhaseSteps"] as? [NSNumber])?
                    .map(\.intValue)
                guard sequence.count == steps.intValue,
                      (denominator ?? 0) > 0,
                      (phaseSteps.map {
                          $0.count == steps.intValue
                            && $0.allSatisfy {
                                $0 >= 0 && $0 < denominator!
                            }
                      } ?? true),
                      sequence.allSatisfy({ !$0.isEmpty
                        && $0.allSatisfy(\.isFinite) }) else { return nil }
                stages.append(.init(
                    trajectory: [], controlSteps: steps.intValue,
                    trajectorySequence: sequence,
                    trajectorySequencePhaseSteps: phaseSteps,
                    trajectorySequenceStepDenominator: denominator,
                    forwardOnlyBaseCommand: forwardOnlyBaseCommand,
                    holonomicBaseCommand: holonomicBaseCommand,
                    locomotionCheckpointDirectory:
                        locomotionCheckpointDirectory,
                    locomotionCommandSpeed: locomotionCommandSpeed,
                    appliedNormalizedActions:
                        appliedNormalizedActions,
                    canonicalizeReplicasBeforeExecution:
                        canonicalizeReplicasBeforeExecution,
                    continueFromPreviousTrajectoryTerminal:
                        continueFromPreviousTrajectoryTerminal,
                    graspAnchorFeedbackBlend:
                        graspAnchorFeedbackBlend,
                    graspAnchorFeedbackVelocityHorizonSeconds:
                        graspAnchorFeedbackVelocityHorizon,
                    graspAnchorFeedbackMaximumActionCorrection:
                        graspAnchorFeedbackMaximumActionCorrection,
                    graspAnchorFeedbackInwardPreloadMeters:
                        graspAnchorFeedbackInwardPreload,
                    leftGraspAnchorBoxLocalMeters: leftGraspAnchor,
                    rightGraspAnchorBoxLocalMeters: rightGraspAnchor,
                    graspAnchorBoxHeightMeters: graspAnchorBoxHeight,
                    minimumCarryDistanceMeters: minimumCarry,
                    minimumDestinationProgressMeters:
                        minimumDestinationProgress,
                    minimumRootDestinationProgressMeters:
                        minimumRootDestinationProgress,
                    minimumTouchdowns:
                        minimumTouchdowns,
                    minimumAlternatingSteps:
                        minimumAlternatingSteps,
                    minimumSwingFootLiftMeters:
                        minimumSwingFootLift,
                    minimumFootAirTimeSeconds:
                        minimumFootAirTime,
                    minimumFootUnloadingFraction:
                        minimumFootUnloading,
                    minimumTerminalFootUnloadingFraction:
                        minimumTerminalFootUnloading,
                    minimumClearanceMeters: minimumClearance,
                    maximumPathDownwardBoxVelocityMPS:
                        maximumPathDownwardBoxVelocity,
                    minimumGraspQuality: minimumGraspQuality,
                    certificationDwellSteps: certificationDwell))
                continue
            }
            guard let values = encoded["trajectory"] as? [NSNumber] else {
                return nil
            }
            let duration = (encoded["trajectoryDurationSteps"]
                as? NSNumber)?.intValue
            let trajectory = values.map(\.floatValue)
            guard !trajectory.isEmpty,
                  trajectory.allSatisfy(\.isFinite) else { return nil }
            stages.append(.init(
                trajectory: trajectory, controlSteps: steps.intValue,
                trajectoryDurationSteps: duration,
                forwardOnlyBaseCommand: forwardOnlyBaseCommand,
                holonomicBaseCommand: holonomicBaseCommand,
                locomotionCheckpointDirectory:
                    locomotionCheckpointDirectory,
                locomotionCommandSpeed: locomotionCommandSpeed,
                appliedNormalizedActions:
                    appliedNormalizedActions,
                canonicalizeReplicasBeforeExecution:
                    canonicalizeReplicasBeforeExecution,
                continueFromPreviousTrajectoryTerminal:
                    continueFromPreviousTrajectoryTerminal,
                graspAnchorFeedbackBlend:
                    graspAnchorFeedbackBlend,
                graspAnchorFeedbackVelocityHorizonSeconds:
                    graspAnchorFeedbackVelocityHorizon,
                graspAnchorFeedbackMaximumActionCorrection:
                    graspAnchorFeedbackMaximumActionCorrection,
                graspAnchorFeedbackInwardPreloadMeters:
                    graspAnchorFeedbackInwardPreload,
                leftGraspAnchorBoxLocalMeters: leftGraspAnchor,
                rightGraspAnchorBoxLocalMeters: rightGraspAnchor,
                graspAnchorBoxHeightMeters: graspAnchorBoxHeight,
                minimumCarryDistanceMeters: minimumCarry,
                minimumDestinationProgressMeters:
                    minimumDestinationProgress,
                minimumRootDestinationProgressMeters:
                    minimumRootDestinationProgress,
                minimumTouchdowns:
                    minimumTouchdowns,
                minimumAlternatingSteps:
                    minimumAlternatingSteps,
                minimumSwingFootLiftMeters:
                    minimumSwingFootLift,
                minimumFootAirTimeSeconds:
                    minimumFootAirTime,
                minimumFootUnloadingFraction:
                    minimumFootUnloading,
                minimumTerminalFootUnloadingFraction:
                    minimumTerminalFootUnloading,
                minimumClearanceMeters: minimumClearance,
                maximumPathDownwardBoxVelocityMPS:
                    maximumPathDownwardBoxVelocity,
                minimumGraspQuality: minimumGraspQuality,
                certificationDwellSteps: certificationDwell))
        }
    }
    if includeOwnStage {
        let declaredCanonicalBoundary = dictionary[
            "derivedStageSourceCanonicalized"] as? Bool
        let inferredCanonicalBoundary: Bool
        if useTargetGeneratingTrajectory {
            inferredCanonicalBoundary = dictionary[
                "targetGeneratingTrajectorySequencePhaseSteps"] != nil
        } else {
            inferredCanonicalBoundary = dictionary[
                "reconstructionValidationCandidateCount"] != nil
        }
        let canonicalizeOwnStage = declaredCanonicalBoundary
            ?? inferredCanonicalBoundary
        let continueOwnStage = dictionary[
            "derivedStageContinuesFromSourceTerminal"] as? Bool == true
        let requiredGraspQuality = ((dictionary[
            "minimumTargetGraspQuality"]
            ?? dictionary["requiredGraspQuality"])
            as? NSNumber)?.floatValue ?? 0
        let requiredExecutionSteps = (dictionary["targetExecutionSteps"]
            as? NSNumber)?.intValue ?? 0
        let measuredGraspDwell = (dictionary[
            "maximumGraspQualityThresholdDwellSteps"]
            as? NSNumber)?.intValue ?? 0
        let qualification = HumanoidBoxFlowQualification(dictionary)
        let ownStageIsBalanceOnly =
            (dictionary["physicalBalanceGatePassed"] as? Bool) == true
            && (requiredGraspQuality <= 0
                || (requiredExecutionSteps > 0
                    && measuredGraspDwell >= requiredExecutionSteps))
            && !qualification.reusableFrontierPassed
        if useTargetGeneratingTrajectory,
           let encodedSequence = dictionary[
                "targetGeneratingTrajectorySequence"] as? [[NSNumber]],
           let denominator = (dictionary["recedingHorizonSteps"]
                ?? dictionary["targetGenerationSteps"]) as? NSNumber {
            let sequence = encodedSequence.map { $0.map(\.floatValue) }
            let phaseSteps = (dictionary[
                "targetGeneratingTrajectorySequencePhaseSteps"]
                as? [NSNumber])?.map(\.intValue)
            let committedActions = humanoidBoxCommittedActions(
                dictionary["targetCommittedTrace"]
                    ?? dictionary["committedTrace"],
                expectedCount: sequence.count)
            guard !sequence.isEmpty, denominator.intValue > 0,
                  !committedActions.present
                    || committedActions.actions != nil,
                  (phaseSteps.map {
                      $0.count == sequence.count
                        && $0.allSatisfy {
                            $0 >= 0 && $0 < denominator.intValue
                        }
                  } ?? true),
                  sequence.allSatisfy({ !$0.isEmpty
                    && $0.allSatisfy(\.isFinite) }) else { return nil }
            stages.append(.init(
                trajectory: [], controlSteps: sequence.count,
                trajectorySequence: sequence,
                trajectorySequencePhaseSteps: phaseSteps,
                trajectorySequenceStepDenominator: denominator.intValue,
                forwardOnlyBaseCommand: dictionary[
                    "recedingForwardOnlyBaseCommand"] as? Bool == true,
                holonomicBaseCommand: dictionary[
                    "recedingHolonomicBaseCommand"] as? Bool == true,
                locomotionCheckpointDirectory: dictionary[
                    "recedingLocomotionCheckpointDirectory"] as? String,
                locomotionCommandSpeed: (dictionary[
                    "recedingLocomotionCommandSpeed"]
                    as? NSNumber)?.floatValue,
                appliedNormalizedActions:
                    committedActions.actions,
                canonicalizeReplicasBeforeExecution:
                    canonicalizeOwnStage,
                continueFromPreviousTrajectoryTerminal:
                    continueOwnStage,
                graspAnchorFeedbackBlend: (dictionary[
                    "graspAnchorFeedbackBlend"] as? NSNumber)?.floatValue,
                graspAnchorFeedbackVelocityHorizonSeconds: (dictionary[
                    "graspAnchorFeedbackVelocityHorizonSeconds"]
                    as? NSNumber)?.floatValue,
                graspAnchorFeedbackMaximumActionCorrection: (dictionary[
                    "graspAnchorFeedbackMaximumActionCorrection"]
                    as? NSNumber)?.floatValue,
                graspAnchorFeedbackInwardPreloadMeters: (dictionary[
                    "graspAnchorFeedbackInwardPreloadMeters"]
                    as? NSNumber)?.floatValue,
                leftGraspAnchorBoxLocalMeters: (dictionary[
                    "leftGraspAnchorBoxLocalMeters"] as? [NSNumber])?
                    .map(\.floatValue),
                rightGraspAnchorBoxLocalMeters: (dictionary[
                    "rightGraspAnchorBoxLocalMeters"] as? [NSNumber])?
                    .map(\.floatValue),
                graspAnchorBoxHeightMeters: (dictionary[
                    "graspAnchorBoxHeightMeters"]
                    as? NSNumber)?.floatValue,
                minimumCarryDistanceMeters: ((dictionary[
                    "minimumTargetCarryDistanceMeters"]
                    ?? dictionary["requiredCarryDistanceMeters"])
                    as? NSNumber)?.floatValue,
                minimumDestinationProgressMeters: ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetDestinationProgressMeters"]
                        ?? dictionary["requiredDestinationProgressMeters"])
                        as? NSNumber)?.floatValue,
                minimumRootDestinationProgressMeters: ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetRootDestinationProgressMeters"]
                        ?? dictionary[
                            "requiredRootDestinationProgressMeters"])
                        as? NSNumber)?.floatValue,
                minimumTouchdowns: ownStageIsBalanceOnly
                    ? nil : ((dictionary["minimumTargetTouchdowns"]
                        ?? dictionary["requiredTouchdowns"])
                        as? NSNumber)?.intValue,
                minimumAlternatingSteps: ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetAlternatingSteps"]
                        ?? dictionary["requiredAlternatingSteps"])
                        as? NSNumber)?.intValue,
                minimumSwingFootLiftMeters: ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetSwingFootLiftMeters"]
                        ?? dictionary["requiredSwingFootLiftMeters"])
                        as? NSNumber)?.floatValue,
                minimumFootAirTimeSeconds: ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetFootAirTimeSeconds"]
                        ?? dictionary["requiredFootAirTimeSeconds"])
                        as? NSNumber)?.floatValue,
                minimumFootUnloadingFraction: ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetFootUnloadingFraction"]
                        ?? dictionary[
                            "requiredFootUnloadingFraction"])
                        as? NSNumber)?.floatValue,
                minimumTerminalFootUnloadingFraction:
                    ownStageIsBalanceOnly
                    ? nil : ((dictionary[
                        "minimumTargetTerminalFootUnloadingFraction"]
                        ?? dictionary[
                            "requiredTerminalFootUnloadingFraction"])
                        as? NSNumber)?.floatValue,
                minimumClearanceMeters: ((dictionary[
                    "minimumTargetClearanceMeters"]
                    ?? dictionary["requiredClearanceMeters"])
                    as? NSNumber)?.floatValue,
                maximumPathDownwardBoxVelocityMPS: ((dictionary[
                    "maximumTargetPathDownwardBoxVelocityMPS"]
                    ?? dictionary[
                        "maximumPathDownwardBoxVelocityMPS"])
                    as? NSNumber)?.floatValue,
                minimumGraspQuality: ((dictionary[
                    "minimumTargetGraspQuality"]
                    ?? dictionary["requiredGraspQuality"])
                    as? NSNumber)?.floatValue,
                certificationDwellSteps: ((dictionary[
                    "targetFeasibilityDwellSteps"]
                    ?? dictionary["requiredFeasibilityDwellSteps"])
                    as? NSNumber)?.intValue))
            return stages
        }
        guard let ownValues = dictionary[
                useTargetGeneratingTrajectory
                    ? "targetGeneratingTrajectory" : "bestTrajectory"]
                as? [NSNumber],
              let ownSteps = dictionary["selectedTargetStep"] as? NSNumber,
              ownSteps.intValue > 0 else { return nil }
        let ownTrajectory = ownValues.map(\.floatValue)
        let committedActions = humanoidBoxCommittedActions(
            dictionary["targetCommittedTrace"]
                ?? dictionary["committedTrace"],
            expectedCount: ownSteps.intValue)
        guard !ownTrajectory.isEmpty,
              ownTrajectory.allSatisfy(\.isFinite),
              !committedActions.present
                || committedActions.actions != nil else { return nil }
        stages.append(.init(
            trajectory: ownTrajectory, controlSteps: ownSteps.intValue,
            trajectoryDurationSteps: useTargetGeneratingTrajectory
                ? (dictionary["targetGenerationSteps"] as? NSNumber)?.intValue
                : (dictionary["candidateTrajectoryDurationSteps"]
                    as? NSNumber)?.intValue,
            appliedNormalizedActions: committedActions.actions,
            canonicalizeReplicasBeforeExecution:
                canonicalizeOwnStage,
            continueFromPreviousTrajectoryTerminal:
                continueOwnStage,
            graspAnchorFeedbackBlend: (dictionary[
                "graspAnchorFeedbackBlend"] as? NSNumber)?.floatValue,
            graspAnchorFeedbackVelocityHorizonSeconds: (dictionary[
                "graspAnchorFeedbackVelocityHorizonSeconds"]
                as? NSNumber)?.floatValue,
            graspAnchorFeedbackMaximumActionCorrection: (dictionary[
                "graspAnchorFeedbackMaximumActionCorrection"]
                as? NSNumber)?.floatValue,
            graspAnchorFeedbackInwardPreloadMeters: (dictionary[
                "graspAnchorFeedbackInwardPreloadMeters"]
                as? NSNumber)?.floatValue,
            leftGraspAnchorBoxLocalMeters: (dictionary[
                "leftGraspAnchorBoxLocalMeters"] as? [NSNumber])?
                .map(\.floatValue),
            rightGraspAnchorBoxLocalMeters: (dictionary[
                "rightGraspAnchorBoxLocalMeters"] as? [NSNumber])?
                .map(\.floatValue),
            graspAnchorBoxHeightMeters: (dictionary[
                "graspAnchorBoxHeightMeters"] as? NSNumber)?.floatValue,
            minimumCarryDistanceMeters: (dictionary[
                "minimumTargetCarryDistanceMeters"]
                as? NSNumber)?.floatValue,
            minimumDestinationProgressMeters: ownStageIsBalanceOnly
                ? nil : (dictionary[
                    "minimumTargetDestinationProgressMeters"]
                    as? NSNumber)?.floatValue,
            minimumRootDestinationProgressMeters: ownStageIsBalanceOnly
                ? nil : (dictionary[
                    "minimumTargetRootDestinationProgressMeters"]
                    as? NSNumber)?.floatValue,
            minimumTouchdowns: ownStageIsBalanceOnly
                ? nil : (dictionary["minimumTargetTouchdowns"]
                    as? NSNumber)?.intValue,
            minimumAlternatingSteps: ownStageIsBalanceOnly
                ? nil : (dictionary["minimumTargetAlternatingSteps"]
                    as? NSNumber)?.intValue,
            minimumSwingFootLiftMeters: ownStageIsBalanceOnly
                ? nil : (dictionary["minimumTargetSwingFootLiftMeters"]
                    as? NSNumber)?.floatValue,
            minimumFootAirTimeSeconds: ownStageIsBalanceOnly
                ? nil : (dictionary[
                    "minimumTargetFootAirTimeSeconds"]
                    as? NSNumber)?.floatValue,
            minimumFootUnloadingFraction: ownStageIsBalanceOnly
                ? nil : (dictionary[
                    "minimumTargetFootUnloadingFraction"]
                    as? NSNumber)?.floatValue,
            minimumTerminalFootUnloadingFraction: ownStageIsBalanceOnly
                ? nil : (dictionary[
                    "minimumTargetTerminalFootUnloadingFraction"]
                    as? NSNumber)?.floatValue,
            minimumClearanceMeters: (dictionary[
                "minimumTargetClearanceMeters"]
                as? NSNumber)?.floatValue,
            maximumPathDownwardBoxVelocityMPS: (dictionary[
                "maximumTargetPathDownwardBoxVelocityMPS"]
                as? NSNumber)?.floatValue,
            minimumGraspQuality: (dictionary[
                "minimumTargetGraspQuality"]
                as? NSNumber)?.floatValue,
            certificationDwellSteps: (dictionary[
                "targetFeasibilityDwellSteps"]
                as? NSNumber)?.intValue))
    }
    return stages.isEmpty ? nil : stages
}

func humanoidBoxTargetGeneratingTrajectory(from data: Data) -> [Float]? {
    humanoidBoxNamedTrajectory(
        from: data, key: "targetGeneratingTrajectory")
}

/// Reads a named finite trajectory from a durable physical-flow artifact.
/// Failure reports intentionally retain several distinct search archives; a
/// continuation must select the archive it means to repair instead of relying
/// on whichever diagnostic happened to be appended last.
func humanoidBoxNamedTrajectory(
    from data: Data, key: String
) -> [Float]? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any],
          let values = dictionary[key] as? [NSNumber],
          let declaredCount = humanoidBoxDeclaredTrajectoryElementCount(
            dictionary),
          values.count == declaredCount else {
        return nil
    }
    let trajectory = values.map(\.floatValue)
    return !trajectory.isEmpty && trajectory.allSatisfy(\.isFinite)
        ? trajectory : nil
}

func humanoidBoxTargetGeneratingAction(
    from data: Data, index: Int
) -> [Float]? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any],
          let encoded = dictionary[
            "targetGeneratingTrajectorySequence"] as? [[NSNumber]],
          encoded.indices.contains(index),
          let declaredCount = humanoidBoxDeclaredTrajectoryElementCount(
            dictionary),
          encoded[index].count == declaredCount else {
        return nil
    }
    let values = encoded[index].map(\.floatValue)
    return !values.isEmpty && values.allSatisfy(\.isFinite)
        ? values : nil
}

func humanoidBoxLastTargetGeneratingAction(from data: Data) -> [Float]? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any],
          let encoded = dictionary[
            "targetGeneratingTrajectorySequence"] as? [[NSNumber]],
          !encoded.isEmpty else {
        return nil
    }
    return humanoidBoxTargetGeneratingAction(
        from: data, index: encoded.count - 1)
}

private struct HumanoidBoxFlowTrajectorySchema {
    let blendKnots: Int
    let residualKnots: Int
    let torsoResidualKnots: Int
    let residualMaximum: Float
    let torsoResidualMaximum: Float
    let armAsymmetryKnots: Int
    let armAsymmetryMaximum: Float

    /// The structured action heads are explicitly declared by the artifact.
    /// Subtracting exactly those heads leaves the four arm splines; divisibility
    /// is validation of that schema, not a guess based on the total length.
    func armParameterCount(totalParameterCount: Int) -> Int? {
        let count = totalParameterCount
            - blendKnots
            - 10 * residualKnots
            - torsoResidualKnots
            - 4 * armAsymmetryKnots
        return count > 0 && count.isMultiple(of: 4) ? count : nil
    }
}

private func humanoidBoxFlowTrajectorySchema(
    from data: Data
) -> HumanoidBoxFlowTrajectorySchema? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any] else { return nil }
    let blend = (dictionary["legBlendKnotCount"] as? NSNumber)?.intValue
        ?? 0
    let residual = (dictionary["legResidualKnotCount"] as? NSNumber)?.intValue
        ?? 0
    let torso = (dictionary["torsoResidualKnotCount"] as? NSNumber)?.intValue
        ?? 0
    let residualMaximum = (dictionary[
        "maximumLegResidualAction"] as? NSNumber)?.floatValue ?? 0.25
    let torsoResidualMaximum = (dictionary[
        "maximumTorsoResidualAction"] as? NSNumber)?.floatValue ?? 0.25
    let armAsymmetry = (dictionary[
        "armAsymmetryKnotCount"] as? NSNumber)?.intValue ?? 0
    let armAsymmetryMaximum = (dictionary[
        "maximumArmAsymmetryAction"] as? NSNumber)?.floatValue ?? 0.25
    guard blend >= 0, residual >= 0, torso >= 0, armAsymmetry >= 0,
          residualMaximum.isFinite, residualMaximum > 0,
          torsoResidualMaximum.isFinite, torsoResidualMaximum > 0,
          armAsymmetryMaximum.isFinite, armAsymmetryMaximum > 0 else {
        return nil
    }
    return HumanoidBoxFlowTrajectorySchema(
        blendKnots: blend, residualKnots: residual,
        torsoResidualKnots: torso,
        residualMaximum: residualMaximum,
        torsoResidualMaximum: torsoResidualMaximum,
        armAsymmetryKnots: armAsymmetry,
        armAsymmetryMaximum: armAsymmetryMaximum)
}

func humanoidBoxFlowSelectedTargetStep(from data: Data) -> Int? {
    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any],
          let step = dictionary["selectedTargetStep"] as? NSNumber,
          step.intValue > 0 else { return nil }
    return step.intValue
}

func parseOptions(_ args: [String]) -> Options {
    var o = Options()
    var i = 0
    func value(after flag: String) -> String {
        guard i + 1 < args.count else { fail("missing value after \(flag)") }
        i += 1
        return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "--preset":
            let name = value(after: args[i])
            o.preset = name
            switch name {
            case "maniskill-pusht-ppo":
                // Official examples/baselines/ppo/baselines.sh and
                // ppo_fast.py profile for PushT-v1 (50M transitions).
                o.envs = 4_096
                o.updates = 762
                o.horizon = 16
                o.epochs = 8
                o.batch = 2_048
                o.lr = 3e-4
                o.optimizerEpsilon = 1e-5
                o.gamma = 0.99
                o.gaeLambda = 0.9
                o.hidden = 256
                o.hiddenLayers = [256, 256, 256]
                o.actionDistribution = .gaussian
                o.actionStd = 1
                o.symmetryAugmentation = false
                o.symmetryMirrorLossCoefficient = 0
                o.normalizeObservations = false
                o.updateObservationNormalizer = false
                o.entropy = 0
                o.targetKL = 0.1
                o.klSchedule = .earlyStop
                o.policyClip = 0.2
                o.clipValueLoss = false
                o.valueCoefficient = 0.5
                o.maxGradientNorm = 0.5
                o.checkpointInterval = 25
                o.activation = .tanh
                o.orthogonalInitialization = true
                o.actorOutputGain = 0.01 * sqrt(2)
                o.seed = 9_351
                o.runName = "ppo-maniskill-reference"
            default:
                fail("unknown training preset '\(name)'; available: maniskill-pusht-ppo")
            }
        case "--frames": o.frames = Int(value(after: args[i])) ?? o.frames
        case "--iterations": o.iterations = Int(value(after: args[i]))
        case "--scale": o.scale = Int(value(after: args[i])) ?? o.scale
        case "--dt": o.dt = Float(value(after: args[i]))
        case "--cpu": o.useCPU = true
        case "--json": o.json = true
        case "--watch": o.watch = Int(value(after: args[i]))
        case "--res": o.res = Int(value(after: args[i]))
        case "--stats-every":
            o.statsEvery = Int(value(after: args[i])) ?? o.statsEvery
        case "--envs": o.envs = Int(value(after: args[i])) ?? o.envs
        case "--batch": o.batch = Int(value(after: args[i])) ?? o.batch
        case "--latent": o.latent = Int(value(after: args[i])) ?? o.latent
        case "--lr": o.lr = Float(value(after: args[i])) ?? o.lr
        case "--adam-epsilon":
            o.optimizerEpsilon = Float(value(after: args[i]))
        case "--gamma": o.gamma = Float(value(after: args[i])) ?? o.gamma
        case "--gae-lambda":
            o.gaeLambda = Float(value(after: args[i])) ?? o.gaeLambda
        case "--reward-scale":
            o.rewardScale = Float(value(after: args[i])) ?? o.rewardScale
        case "--lambda": o.lambda = Float(value(after: args[i])) ?? o.lambda
        case "--episodes": o.episodes = Int(value(after: args[i])) ?? o.episodes
        case "--updates": o.updates = Int(value(after: args[i])) ?? o.updates
        case "--horizon": o.horizon = Int(value(after: args[i])) ?? o.horizon
        case "--epochs": o.epochs = Int(value(after: args[i])) ?? o.epochs
        case "--hidden": o.hidden = Int(value(after: args[i])) ?? o.hidden
        case "--hidden-layers":
            let widths = value(after: args[i]).split(separator: ",").compactMap {
                Int($0)
            }
            guard widths.count == 3 else {
                fail("--hidden-layers expects three comma-separated integers")
            }
            o.hiddenLayers = widths
        case "--action-distribution":
            let name = value(after: args[i])
            guard let distribution = PPOActionDistribution(rawValue: name) else {
                fail("--action-distribution must be gaussian or squashed-gaussian")
            }
            o.actionDistribution = distribution
        case "--seed": o.seed = UInt64(value(after: args[i])) ?? o.seed
        case "--algorithm": o.algorithm = value(after: args[i])
        case "--action-std":
            o.actionStd = Float(value(after: args[i])) ?? o.actionStd
        case "--minimum-action-std":
            o.minimumActionStd = Float(value(after: args[i]))
        case "--maximum-action-std":
            o.maximumActionStd = Float(value(after: args[i]))
        case "--final-action-std":
            o.finalActionStd = Float(value(after: args[i]))
        case "--action-std-anneal-start":
            o.actionStdAnnealStartUpdate = Int(value(after: args[i]))
        case "--action-std-anneal-end":
            o.actionStdAnnealEndUpdate = Int(value(after: args[i]))
        case "--initialize-from":
            o.initializeFrom = value(after: args[i])
        case "--policy-expert-from":
            o.policyExpertFrom = value(after: args[i])
        case "--policy-expert-branch-from":
            o.policyExpertBranchFrom = value(after: args[i])
        case "--stand-expert-from":
            o.standExpertFrom = value(after: args[i])
        case "--locomotion-checkpoint":
            o.locomotionCheckpoint = value(after: args[i])
        case "--initialization-normalizer-prior-count":
            o.initializationNormalizerPriorCount =
                Double(value(after: args[i]))
        case "--no-symmetry-augmentation": o.symmetryAugmentation = false
        case "--symmetry-mirror-loss":
            o.symmetryMirrorLossCoefficient = Float(value(after: args[i]))
        case "--legacy-symmetry-data-augmentation":
            o.symmetryMirrorLossCoefficient = 0
        case "--freeze-observation-normalizer":
            o.updateObservationNormalizer = false
        case "--no-observation-normalization":
            o.normalizeObservations = false
        case "--entropy": o.entropy = Float(value(after: args[i])) ?? o.entropy
        case "--target-kl": o.targetKL = Float(value(after: args[i])) ?? o.targetKL
        case "--kl-schedule":
            let name = value(after: args[i])
            guard let schedule = PPOKLSchedule(rawValue: name) else {
                fail("--kl-schedule must be adaptive, early-stop, or none")
            }
            o.klSchedule = schedule
        case "--policy-clip":
            o.policyClip = Float(value(after: args[i])) ?? o.policyClip
        case "--value-clip":
            o.valueClip = Float(value(after: args[i])) ?? o.valueClip
        case "--no-value-clip-loss": o.clipValueLoss = false
        case "--value-coef":
            o.valueCoefficient = Float(value(after: args[i]))
                ?? o.valueCoefficient
        case "--max-grad-norm":
            o.maxGradientNorm = Float(value(after: args[i]))
                ?? o.maxGradientNorm
        case "--success-imitation-coef":
            o.successImitationCoefficient = Float(value(after: args[i]))
                ?? o.successImitationCoefficient
        case "--success-replay-capacity":
            o.successReplayCapacity = Int(value(after: args[i]))
                ?? o.successReplayCapacity
        case "--success-replay-batch":
            o.successReplayBatchSize = Int(value(after: args[i]))
                ?? o.successReplayBatchSize
        case "--success-imitation-history":
            o.successImitationHistorySteps = Int(value(after: args[i]))
        case "--reference-policy-coef":
            o.referencePolicyCoefficient = Float(value(after: args[i]))
                ?? o.referencePolicyCoefficient
        case "--checkpoint-interval":
            o.checkpointInterval = Int(value(after: args[i]))
                ?? o.checkpointInterval
        case "--activation":
            let name = value(after: args[i])
            guard let activation = PPOActivation(rawValue: name) else {
                fail("--activation must be elu or tanh")
            }
            o.activation = activation
        case "--orthogonal-initialization":
            o.orthogonalInitialization = true
        case "--actor-output-gain":
            o.actorOutputGain = Float(value(after: args[i]))
        case "--resume": o.resume = true
        case "--run": o.runName = value(after: args[i])
        case "--checkpoint": o.checkpoint = value(after: args[i])
        case "--output": o.output = value(after: args[i])
        case "--allow-task-transfer": o.allowTaskTransfer = true
        case "--expected-checkpoint-fingerprint":
            o.expectedCheckpointFingerprint = value(after: args[i])
        case "--source-commit": o.sourceCommit = value(after: args[i])
        case "--inference-batch-size":
            o.inferenceBatchSize = Int(value(after: args[i]))
        case "--evaluation-seeds":
            let rawSeeds = value(after: args[i]).split(
                separator: ",", omittingEmptySubsequences: false)
            let seeds = rawSeeds.compactMap { UInt64($0) }
            guard rawSeeds.count == 4, seeds.count == 4 else {
                fail("--evaluation-seeds expects four comma-separated UInt64 values")
            }
            o.evaluationSeeds = seeds
        case "--evaluation-environments":
            o.evaluationEnvironments = Int(value(after: args[i]))
        case "--parent-checkpoint":
            o.parentCheckpoint = value(after: args[i])
        case "--qualification-directory":
            o.qualificationDirectory = value(after: args[i])
        case "--task-option":
            let assignment = value(after: args[i])
            let pieces = assignment.split(separator: "=", maxSplits: 1,
                                          omittingEmptySubsequences: false)
            guard pieces.count == 2, !pieces[0].isEmpty,
                  let numericValue = Float(pieces[1]) else {
                fail("--task-option expects key=value with a numeric value")
            }
            o.taskOptions[String(pieces[0])] = numericValue
        case "--gait-swing-steps":
            o.gaitSwingSteps = Int(value(after: args[i]))
                ?? o.gaitSwingSteps
        case "--gait-swing-height":
            o.gaitSwingHeight = Float(value(after: args[i]))
                ?? o.gaitSwingHeight
        case "--gait-placement-horizon":
            o.gaitPlacementHorizon = Float(value(after: args[i]))
                ?? o.gaitPlacementHorizon
        case "--gait-maximum-placement":
            o.gaitMaximumPlacement = Float(value(after: args[i]))
                ?? o.gaitMaximumPlacement
        default: break
        }
        i += 1
    }
    return o
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("error: " + msg + "\n").data(using: .utf8)!)
    exit(1)
}

func makeScene(_ name: String, _ o: Options) -> PhysicsScene {
    guard var scene = Demos.make(name, scale: o.scale, res: o.res) else {
        fail("unknown demo '\(name)'. Available: \(Demos.all.joined(separator: ", "))")
    }
    if let it = o.iterations { scene.settings.iterations = it }
    if let dt = o.dt { scene.settings.dt = dt }
    return scene
}

/// Reconstruct the exact serialized task options while resolving its current
/// registered physics revision. Requalification may change that revision and
/// nothing else.
func makeCurrentCheckpointTask(
    taskID: String, checkpointDirectory: String, seed: UInt64 = 1
) throws -> any VectorizedRLTask {
    let metadata = try JSONDecoder().decode(
        VectorPolicyMetadata.self,
        from: Data(contentsOf: URL(
            fileURLWithPath: "\(checkpointDirectory)/metadata.json")))
    let options = BuiltInRLTasks.registry.checkpointReplayOptions(
        for: taskID,
        semanticOptions: metadata.taskConfiguration ?? [:],
        maxEpisodeSteps: metadata.maxEpisodeSteps,
        controlDecimation: metadata.controlDecimation)
    return try BuiltInRLTasks.registry.make(
        taskID, configuration: RLTaskConfiguration(
            numEnvironments: 1, seed: seed, autoReset: false,
            options: options))
}

func makePushTPhysicalFlowConfiguration(
    _ o: Options, arguments: [String], defaultGenerations: Int = 6
) -> PushTPhysicalFlowConfiguration {
    let endpointContact = o.algorithm.hasPrefix("endpoint-contact")
    let fullCovariance = o.algorithm.contains("full-cem")
    let balancedObjective = o.algorithm.hasSuffix("balanced")
    return PushTPhysicalFlowConfiguration(
        populationSize: max(8, o.envs),
        simulationBatchSize: max(
            3, min(32, Int(o.taskOptions["simulationBatchSize"] ?? 32))),
        generations: max(1, o.iterations ?? defaultGenerations),
        horizon: arguments.contains("--horizon") ? max(8, o.horizon) : 48,
        controlPointCount: max(
            4, Int(o.taskOptions["controlPointCount"] ?? 6)),
        substeps: max(1, Int(o.taskOptions["substeps"] ?? 4)),
        sourcePreparationSteps: max(
            1, Int(o.taskOptions["sourcePreparationSteps"] ?? 160)),
        targetSettlingSteps: max(
            0, Int(o.taskOptions["targetSettlingSteps"] ?? 0)),
        terminalHoldSteps: max(
            0, Int(o.taskOptions["terminalHoldSteps"] ?? 0)),
        targetGenerationMaximumStep:
            o.taskOptions["targetGenerationMaximumStep"] ?? 0.16,
        proposal: endpointContact ? .endpointContact : .linearEndpoints,
        covarianceMode: fullCovariance ? .full : .diagonal,
        objective: balancedObjective
            ? .balancedEndpointBottleneck : .legacyWeightedState,
        initialStandardDeviation:
            o.taskOptions["initialStandardDeviation"] ?? 0.65,
        eliteFraction: o.taskOptions["eliteFraction"] ?? 0.1,
        seed: o.seed)
}

setvbuf(stdout, nil, _IOLBF, 0)   // line-buffer even when redirected to a log

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    print("""
    avbd — Augmented Vertex Block Descent (Metal)

    Commands:
      run <demo>     Simulate a demo scene headlessly
      bench <demo>   Benchmark ms/frame
      parity <demo>  Compare GPU vs CPU trajectories
      list           List demo scenes
      list-rl        List vectorized robot-learning tasks and algorithms
      describe-rl <task>  Show accepted task options, types, and bounds
      rl-smoke <task>  Step a vectorized task and validate finite tensors
      eval-arachne-classical  Evaluate non-neural ripple-gait point-goal control
      train-rl <task>  Train any registered task with MLX PPO
      eval-rl <task>   Evaluate a saved MLX policy deterministically
      prepare-policy-requalification <task>  Create a zero-update revision candidate
      publish-policy-requalification <task>  Seal a robustly evaluated candidate
      verify-policy-requalification <task>  Verify sealed revision lineage/evidence
      export-policy-rl <task>  Create an immutable optimizer-free field bundle
      verify-policy-rl <task>  Verify immutable bundle identity and MLX inference
      trace-rl <task>  Trace one deterministic policy trajectory step by step
      distill-h1-box-lift  Distill a verified physical lift trajectory into MLX
      train-h1-box-action-chunk  Fit a zero-preserving frontier action chunk
      eval-h1-box-action-chunk  Replay immutable chunk weights in exact physics
      interpolate-h1-box-flow  Physically select a policy residual scale
      probe-h1-box-lift  Audit bounded H1 grasp-to-lift actuation in parallel
      experiment-h1-box-flow  Reconstruct a hidden physical H1+box future
      eval-arm-expert  Evaluate the batched Push-T demonstration expert
      experiment-pusht-flow  Reconstruct a real Push-T future with spline CEM
      eval-pusht-flow  Evaluate geometric physical-flow search over many seeds
      collect-pusht-flow-teacher  Build physically verified inverse-flow rows
      train-pusht-flow-proposal  Train one structured MLX spline proposal net
      eval-pusht-flow-proposal  Evaluate learned proposals in exact physics
      eval-pusht-flow-retrieval  Evaluate nearest verified spline proposals
      select-rl <reports...>  Select one checkpoint on validation-only reports
      verify-selection-rl <selection.json> <reports...>  Reject seed leakage or drift
      aggregate-rl <reports...>  Aggregate distinct-seed evaluation JSON files
      aggregate-checkpoint-rl <reports...>  Aggregate one checkpoint over reset seeds
      sim2sim-h1       Run Unitree's imported recurrent H1 policy unchanged
      sim2sim-gear-sonic <reference-dir>  Replay NVIDIA GEAR-SONIC on G1

    Options: --frames N --iterations N --scale N --dt T --cpu --json --watch BODY --stats-every N
    """)
    exit(0)
}

switch command {
case "list":
    for d in Demos.all { print(d) }

case "list-rl":
    print("tasks:")
    for id in BuiltInRLTasks.registry.taskIDs { print("  \(id)") }
    print("algorithms:")
    for id in VectorRLAlgorithmRegistry.builtIn.algorithmIDs { print("  \(id)") }

case "describe-rl":
    guard args.count > 1 else {
        fail("usage: avbd describe-rl <task> [--json]")
    }
    let taskID = args[1]
    guard BuiltInRLTasks.registry.taskIDs.contains(taskID) else {
        fail("unknown RL task '\(taskID)'; available: "
            + BuiltInRLTasks.registry.taskIDs.joined(separator: ", "))
    }
    guard let schema = BuiltInRLTasks.registry.optionSchema(for: taskID) else {
        print("\(taskID) has no registered option schema")
        break
    }
    if args.contains("--json") {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(schema), as: UTF8.self))
        break
    }
    print("\(taskID) options:")
    for name in schema.optionNames {
        let definition = schema.definitions[name]!
        var constraints = [definition.valueKind.rawValue]
        if let lower = definition.lowerBound {
            constraints.append("min=\(lower)")
        }
        if let upper = definition.upperBound {
            constraints.append("max=\(upper)")
        }
        print("  \(name): \(constraints.joined(separator: ", "))")
    }

case "eval-arachne-classical":
    let o = parseOptions(Array(args.dropFirst(1)))
    let environmentCount = max(1, o.envs)
    let registered = try BuiltInRLTasks.registry.make(
        "arachne15-goal-v0", configuration: RLTaskConfiguration(
            numEnvironments: environmentCount, seed: o.seed,
            autoReset: false, options: o.taskOptions))
    guard let task = registered as? Arachne15LocomotionTask else {
        fail("arachne15-goal-v0 did not construct Arachne15LocomotionTask")
    }
    _ = try task.reset(seed: o.seed)
    let initialStates = task.environment.states()
    let controller = Arachne15ClassicalController(configuration: .init(
        swingSteps: o.gaitSwingSteps,
        swingHeight: o.gaitSwingHeight,
        placementHorizon: o.gaitPlacementHorizon,
        maximumPlanarPlacement: o.gaitMaximumPlacement))
    controller.reset(states: initialStates)
    var result = RLStepBatch(spec: task.spec)
    var finished = [Bool](repeating: false, count: environmentCount)
    var successCount = 0
    var survivalCount = 0
    var completionSteps = [Int](repeating: task.spec.maxEpisodeSteps,
                                count: environmentCount)
    var finalDistances = [Float](repeating: .nan, count: environmentCount)
    var minimumDistances = [Float](repeating: .nan, count: environmentCount)
    var clearances = [Float](repeating: .nan, count: environmentCount)
    var penetrationRMSE = [Float](repeating: .nan, count: environmentCount)
    let start = Date()
    var executedSteps = 0
    var constrainedTargetCount = 0
    for step in 0..<task.spec.maxEpisodeSteps where finished.contains(false) {
        executedSteps += 1
        let states = task.environment.states()
        let commands = (0..<environmentCount).map {
            finished[$0] ? F3.zero : task.currentCommand(environment: $0)
        }
        let actions = controller.actions(
            states: states, commands: commands, spec: task.spec)
        constrainedTargetCount += controller.diagnostics.constrainedTargetCount
        try task.step(actions: actions, into: &result)
        if args.contains("--trace") && (step + 1).isMultiple(of: o.statsEvery) {
            let state = task.environment.states()[0]
            let command = task.currentCommand(environment: 0)
            let localVelocity = state.root.rotation.conjugate.act(
                state.root.linearVelocity)
            let distance = task.currentGoalDistance(environment: 0)
            let contacts = task.environment.groundContacts()[0]
                .filter { $0 }.count
            let up = state.root.rotation.conjugate.act(F3(0, 0, 1)).z
            let meanAction = actions.values.prefix(16).map(abs).reduce(0, +)
                / 16
            print(String(format:
                "step %4d goal %.3f root (%+.3f,%+.3f,%.3f) "
                    + "velocity (%+.3f,%+.3f) command (%+.3f,%+.3f,%+.3f) "
                    + "up %.3f contacts %d action %.3f constraints %d",
                step + 1, distance, state.root.position.x,
                state.root.position.y, state.root.position.z,
                localVelocity.x, localVelocity.y, command.x, command.y,
                command.z, up, contacts, meanAction,
                controller.diagnostics.constrainedTargetCount))
        }
        for e in 0..<environmentCount where !finished[e]
            && (result.terminated[e] || result.truncated[e]) {
            finished[e] = true
            completionSteps[e] = step + 1
            if result.successes[e] { successCount += 1 }
            if (result.metrics["episode/survived"]?[e] ?? 0) > 0.5 {
                survivalCount += 1
            }
            finalDistances[e] = result.metrics[
                "episode/final_goal_distance_m"]?[e] ?? .nan
            minimumDistances[e] = result.metrics[
                "episode/minimum_goal_distance_m"]?[e] ?? .nan
            clearances[e] = result.metrics[
                "episode/minimum_foot_collider_clearance_m"]?[e] ?? .nan
            penetrationRMSE[e] = result.metrics[
                "episode/foot_collider_penetration_rmse_m"]?[e] ?? .nan
        }
    }
    func finiteMean(_ values: [Float]) -> Float {
        let finite = values.filter(\.isFinite)
        return finite.isEmpty ? .nan : finite.reduce(0, +) / Float(finite.count)
    }
    let elapsed = Date().timeIntervalSince(start)
    let report: [String: Any] = [
        "controller": "six-support-leg-paired-ripple-cpg-ik",
        "controllerConfiguration": [
            "swingSteps": controller.configuration.swingSteps,
            "swingHeightM": controller.configuration.swingHeight,
            "placementHorizonS": controller.configuration.placementHorizon,
            "maximumPlanarPlacementM":
                controller.configuration.maximumPlanarPlacement,
            "minimumTranslationSpeedMPS":
                controller.configuration.minimumTranslationSpeed,
            "standingCommandThreshold":
                controller.configuration.standingCommandThreshold,
        ],
        "task": task.spec.id,
        "taskRevision": task.spec.revision,
        "taskConfiguration": task.spec.configurationValues,
        "seed": o.seed,
        "episodes": environmentCount,
        "completedEpisodes": finished.filter { $0 }.count,
        "successRate": Float(successCount) / Float(environmentCount),
        "survivalRate": Float(survivalCount) / Float(environmentCount),
        "meanCompletionSteps": finiteMean(completionSteps.map(Float.init)),
        "meanFinalGoalDistanceM": finiteMean(finalDistances),
        "meanMinimumGoalDistanceM": finiteMean(minimumDistances),
        "meanPlanarDisplacementM": finiteMean(zip(
            initialStates, task.environment.states()).map { initial, final in
                let delta = final.root.position - initial.root.position
                return sqrt(delta.x * delta.x + delta.y * delta.y)
            }),
        "meanMinimumFootColliderClearanceM": finiteMean(clearances),
        "meanFootPenetrationRMSEM": finiteMean(penetrationRMSE),
        "elapsedSeconds": elapsed,
        "simulatedStepsPerSecond": elapsed > 0
            ? Double(environmentCount * executedSteps) / elapsed : 0,
        "constrainedTargetFraction": Float(constrainedTargetCount)
            / Float(max(executedSteps * environmentCount * 8, 1)),
    ]
    let reportData = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try reportData.write(to: url, options: .atomic)
    }
    print(String(decoding: reportData, as: UTF8.self))

case "sim2sim-h1":
    let o = parseOptions(Array(args.dropFirst(1)))
    let policyDirectory = o.checkpoint ?? "checkpoints/external/unitree-h1"
    let command = SIMD3<Float>(
        o.taskOptions["forward"] ?? 0.5,
        o.taskOptions["lateral"] ?? 0,
        o.taskOptions["yaw"] ?? 0)
    let session = try UnitreeH1Sim2SimSession(
        policyDirectory: policyDirectory, command: command,
        solverIterations: o.iterations)
    let report = try session.run(controlSteps: o.frames) { step, state in
        if args.contains("--trace") {
            func values(_ input: some Collection<Float>) -> String {
                input.map { String(format: "%+.7f", $0) }
                    .joined(separator: ",")
            }
            let rootValues: [Float] = [
                state.root.position.x, state.root.position.y,
                state.root.position.z,
            ]
            print("trace \(step) root [\(values(rootValues))] "
                + "q [\(values(state.jointAngles.prefix(10)))] "
                + "dq [\(values(state.jointVelocities.prefix(10)))] "
                + "obs [\(values(session.lastObservation))] "
                + "action [\(values(session.previousAction))]")
        }
        guard !o.json, step == 1 || step % max(o.statsEvery, 1) == 0 else {
            return
        }
        let upright = state.root.rotation.act(F3(0, 0, 1)).z
        let meanAbsoluteAction = session.previousAction.map(abs).reduce(0, +)
            / Float(session.previousAction.count)
        print(String(format:
            "step %4d  t %6.2f  root (%+.3f,%+.3f,%.3f)  "
            + "upright %.3f  |action| %.3f",
            step, session.elapsedTime, state.root.position.x,
            state.root.position.y, state.root.position.z, upright,
            meanAbsoluteAction))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let encoded = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try encoded.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: encoded, as: UTF8.self))
    } else {
        print(String(format:
            "sim2sim H1  %.2fs  forward %+.3fm (%.3fm/s)  "
            + "lateral %+.3fm  min height %.3fm  min upright %.3f  %@",
            report.simulatedSeconds, report.forwardDistanceMeters,
            report.meanForwardSpeedMetersPerSecond,
            report.lateralDistanceMeters, report.minimumPelvisHeightMeters,
            report.minimumUprightAlignment,
            report.fell
                ? String(format: "FELL at %.2fs",
                         report.firstFallTimeSeconds ?? .nan)
                : "STABLE"))
        print(String(format:
            "TorchScript->MLX max errors: action %.3g  hidden %.3g  cell %.3g  %@",
            report.policyVerification.maximumActionError,
            report.policyVerification.maximumHiddenStateError,
            report.policyVerification.maximumCellStateError,
            report.policyVerification.passed ? "PASS" : "FAIL"))
        print("source sha256 \(report.checkpointSHA256)")
    }
    if !report.finite || !report.policyVerification.passed { exit(2) }

case "sim2sim-gear-sonic":
    guard args.count > 1, !args[1].hasPrefix("--") else {
        fail("usage: avbd sim2sim-gear-sonic <reference-directory> "
            + "[--checkpoint bundle --frames N --envs N --iterations N "
            + "--stats-every N --trace --json --output report.json]")
    }
    let referenceDirectory = args[1]
    let o = parseOptions(Array(args.dropFirst(2)))
    let bundleDirectory = o.checkpoint ?? "checkpoints/external/gear-sonic-g1"
    // `Options.envs` defaults to the large training batch. A replay is a
    // single visible trajectory unless the caller explicitly requests a
    // throughput batch with `--envs`.
    let replayEnvironmentCount = args.contains("--envs") ? max(o.envs, 1) : 1
    let session = try GEARSonicG1Session(
        bundleDirectory: bundleDirectory,
        referenceDirectory: referenceDirectory,
        environmentCount: replayEnvironmentCount,
        solverIterations: o.iterations ?? 8)
    let requestedSteps: Int? = args.contains("--frames") ? o.frames : nil
    let report = try session.run(maximumControlSteps: requestedSteps) {
        step, states in
        let state = states[0]
        if args.contains("--trace") {
            func values(_ input: some Collection<Float>) -> String {
                input.map { String(format: "%+.7f", $0) }
                    .joined(separator: ",")
            }
            let root: [Float] = [
                state.root.position.x, state.root.position.y,
                state.root.position.z,
            ]
            print("trace \(step) frame \(session.referenceFrame) "
                + "root [\(values(root))] "
                + "q [\(values(state.jointAngles))] "
                + "dq [\(values(state.jointVelocities))] "
                + "action [\(values(session.lastRawAction.prefix(29)))]")
        }
        guard !o.json,
              step == 1 || step % max(o.statsEvery, 1) == 0 else { return }
        let upright = state.root.rotation.act(F3(0, 0, 1)).z
        let displayedAction = session.lastRawAction.prefix(29)
        let meanAbsoluteAction = displayedAction.isEmpty ? 0
            : displayedAction.map(abs).reduce(0, +)
                / Float(displayedAction.count)
        print(String(format:
            "step %4d/%4d  t %6.2f  root (%+.3f,%+.3f,%.3f)  "
            + "upright %.3f  |action| %.3f",
            step, session.reference.frameCount, session.elapsedTime,
            state.root.position.x, state.root.position.y,
            state.root.position.z, upright, meanAbsoluteAction))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let encoded = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try encoded.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: encoded, as: UTF8.self))
    } else {
        let actual = report.actualRootDisplacementMeters
        let reference = report.referenceRootDisplacementMeters
        print(String(format:
            "GEAR-SONIC %@  %.2fs (%d/%d frames)  "
            + "root delta (%+.3f,%+.3f,%+.3f)m  "
            + "reference (%+.3f,%+.3f,%+.3f)m",
            report.referenceName, report.simulatedSeconds,
            report.controlSteps, report.referenceFrames,
            actual[0], actual[1], actual[2],
            reference[0], reference[1], reference[2]))
        print(String(format:
            "joint tracking mean %.4frad  max %.4frad  "
            + "min height %.3fm  min upright %.3f  %@",
            report.meanJointTrackingErrorRadians,
            report.maximumJointTrackingErrorRadians,
            report.minimumRootHeightMeters,
            report.minimumUprightAlignment,
            report.firstFallStep.map { "FELL at step \($0)" } ?? "STABLE"))
        print(String(format:
            "14-link root-relative error mean %.4fm  max %.4fm  "
            + "max height error %.4fm  root rotation %.4frad",
            report.meanRootRelativeBodyTrackingErrorMeters,
            report.maximumRootRelativeBodyTrackingErrorMeters,
            report.maximumTrackedBodyHeightErrorMeters,
            report.maximumRootOrientationErrorRadians))
        print(String(format:
            "joint speed max %.3frad/s (%.3fx source limit)  %@",
            report.maximumAbsoluteJointVelocityRadiansPerSecond,
            report.maximumJointVelocityLimitRatio,
            report.firstSourceCriterionFailureStep.map {
                "SOURCE THRESHOLD EXCEEDED at step \($0)"
            } ?? "SOURCE THRESHOLDS PASS"))
        print(String(format:
            "source eval height max %.4fm / %.2fm  orientation %.4frad / %.2frad  "
                + "velocity clamp %@",
            report.maximumSourceCriterionHeightErrorMeters,
            report.sourceHeightFailureThresholdMeters,
            report.maximumRootOrientationErrorRadians,
            report.sourceOrientationFailureThresholdRadians,
            report.jointVelocityLimitsEnforced ? "ENFORCED" : "AUDIT ONLY"))
        print(String(format:
            "ONNX->MLX max errors: token %.3g  action %.3g  %@",
            report.policyMaximumTokenError,
            report.policyMaximumActionError,
            report.policyParityPassed ? "PASS" : "FAIL"))
    }
    if !report.finite || !report.policyParityPassed { exit(2) }
    if report.completedReference && report.firstFallStep != nil { exit(3) }

case "rl-smoke":
    guard args.count > 1 else { fail("usage: avbd rl-smoke <task> [--envs N --frames N]") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let task = try BuiltInRLTasks.registry.make(
        args[1], configuration: RLTaskConfiguration(
            numEnvironments: o.envs, seed: o.seed, options: o.taskOptions))
    var observation = try task.reset(seed: o.seed)
    var actions = RLActionBatch(spec: task.spec)
    var smokeRNG = SplitMix64(seed: o.seed &+ 0xD1B54A32D192ED03)
    func normalSample() -> Float {
        let u1 = max(smokeRNG.nextFloat(), 1e-7)
        let u2 = smokeRNG.nextFloat()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
    var result = RLStepBatch(spec: task.spec)
    let t0 = Date()
    var rewardSum: Float = 0
    var diagnosticMetricSums = [String: Float]()
    var completedMetricSums = [String: Float]()
    var completedEpisodes = 0, successfulEpisodes = 0
    var completedLength: Float = 0
    var completedDistance: Float = 0
    for frame in 0..<o.frames {
        if args.contains("--action-std") {
            let lowerBounds = task.spec.action.lowerBound
            let upperBounds = task.spec.action.upperBound
            for i in actions.values.indices {
                let component = i % actions.actionDimension
                let sample = normalSample() * o.actionStd
                if let lowerBounds, let upperBounds {
                    actions.values[i] = simd_clamp(
                        sample, lowerBounds[component], upperBounds[component])
                } else {
                    actions.values[i] = sample
                }
            }
        }
        try task.step(actions: actions, into: &result)
        if args.contains("--trace") {
            for e in 0..<task.spec.numEnvironments {
                let height = result.metrics["state/torso_height_m"]?[e]
                    ?? .nan
                let up = result.metrics["state/projected_gravity_z"]?[e]
                    ?? .nan
                let margin = result.metrics[
                    "state/minimum_joint_limit_margin_rad"]?[e] ?? .nan
                let clearance = result.metrics[
                    "state/minimum_foot_clearance_m"]?[e] ?? .nan
                let contacts = result.metrics["state/feet_in_contact"]?[e]
                    ?? .nan
                let speed = result.metrics["state/root_planar_speed_mps"]?[e]
                    ?? .nan
                let torqueRatio = result.metrics[
                    "state/maximum_actuator_torque_ratio"]?[e] ?? .nan
                let saturated = result.metrics[
                    "state/saturated_actuator_count"]?[e] ?? .nan
                print(String(format:
                    "frame %d env %d height %.6f up %.6f margin %.6f "
                    + "clearance %.6f contacts %.0f speed %.6f "
                    + "maxTorqueRatio %.3f saturated %.0f "
                    + "terminated %@ truncated %@",
                    frame + 1, e, height, up, margin, clearance, contacts,
                    speed, torqueRatio, saturated,
                    result.terminated[e] ? "true" : "false",
                    result.truncated[e] ? "true" : "false"))
            }
        }
        observation = result.observations
        rewardSum += result.rewards.reduce(0, +)
        for (name, values) in result.metrics
            where name.hasPrefix("reward/") || name.hasPrefix("penalty/")
                || name.hasPrefix("gait/") || name.hasPrefix("state/")
                || name.hasPrefix("task/") {
            diagnosticMetricSums[name, default: 0] += values.reduce(0, +)
        }
        if let lengths = result.metrics["episode/length"] {
            for e in 0..<task.spec.numEnvironments where lengths[e] > 0 {
                completedEpisodes += 1
                completedLength += lengths[e]
                completedDistance += result.metrics["episode/forward_distance_m"]?[e] ?? 0
                for (name, values) in result.metrics
                    where name.hasPrefix("episode/") && e < values.count {
                    completedMetricSums[name, default: 0] += values[e]
                }
                if result.successes[e] { successfulEpisodes += 1 }
            }
        }
        guard observation.policy.allSatisfy(\.isFinite),
              result.rewards.allSatisfy(\.isFinite) else {
            fail("non-finite task tensor")
        }
    }
    let elapsed = max(-t0.timeIntervalSinceNow, 1e-9)
    let transitions = Double(o.frames * task.spec.numEnvironments)
    print(String(format:
        "%@  envs %d  obs %d  act %d  %.0f transitions/s  mean reward %+.4f  "
        + "episodes %d  success %.1f%%  mean length %.1f  mean dx %+.3f m",
        task.spec.id, task.spec.numEnvironments, task.spec.observation.elementCount,
        task.spec.action.elementCount, transitions / elapsed,
        rewardSum / Float(max(o.frames * task.spec.numEnvironments, 1)),
        completedEpisodes,
        completedEpisodes > 0
            ? Float(successfulEpisodes) / Float(completedEpisodes) * 100 : 0,
        completedEpisodes > 0 ? completedLength / Float(completedEpisodes) : 0,
        completedEpisodes > 0 ? completedDistance / Float(completedEpisodes) : 0))
    for (name, total) in diagnosticMetricSums.sorted(by: { $0.key < $1.key }) {
        print(String(format: "  %@ mean %+.6f", name,
                     total / Float(max(o.frames * task.spec.numEnvironments, 1))))
    }
    for (name, total) in completedMetricSums.sorted(by: { $0.key < $1.key }) {
        print(String(format: "  %@ completed-episode mean %+.6f", name,
                     total / Float(max(completedEpisodes, 1))))
    }

case "train-rl":
    guard args.count > 1 else {
        fail("usage: avbd train-rl <task> [--algorithm ppo --envs N --updates N]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let taskID = args[1]
    if o.preset == "maniskill-pusht-ppo"
        && taskID != "maniskill-pusht-v1" {
        fail("maniskill-pusht-ppo preset requires task maniskill-pusht-v1")
    }
    let task = try BuiltInRLTasks.registry.make(
        taskID, configuration: RLTaskConfiguration(
            numEnvironments: o.envs, seed: o.seed, options: o.taskOptions))
    guard o.algorithm == "ppo" else {
        fail("algorithm '\(o.algorithm)' is not configured; available: "
             + VectorRLAlgorithmRegistry.builtIn.algorithmIDs.joined(separator: ", "))
    }
    let config = VectorPPOConfig(
        updates: o.updates, rolloutSteps: o.horizon,
        updateEpochs: o.epochs, minibatchSize: o.batch,
        learningRate: o.lr, optimizerEpsilon: o.optimizerEpsilon,
        gamma: o.gamma, gaeLambda: o.gaeLambda,
        rewardScale: o.rewardScale,
        hiddenSize: o.hidden, seed: o.seed)
    var tunedConfig = config
    tunedConfig.policyClip = o.policyClip
    tunedConfig.valueClip = o.valueClip
    tunedConfig.clipValueLoss = o.clipValueLoss
    tunedConfig.valueCoefficient = o.valueCoefficient
    tunedConfig.maxGradientNorm = o.maxGradientNorm
    tunedConfig.successImitationCoefficient =
        o.successImitationCoefficient
    tunedConfig.successReplayCapacity = o.successReplayCapacity
    tunedConfig.successReplayBatchSize = o.successReplayBatchSize
    tunedConfig.successImitationHistorySteps =
        o.successImitationHistorySteps
    tunedConfig.referencePolicyCoefficient =
        o.referencePolicyCoefficient
    tunedConfig.initialActionStd = o.actionStd
    tunedConfig.hiddenDimensions = o.hiddenLayers
    tunedConfig.activation = o.activation
    tunedConfig.orthogonalInitialization = o.orthogonalInitialization
    tunedConfig.actorOutputGain = o.actorOutputGain
    tunedConfig.actionDistribution = o.actionDistribution
    tunedConfig.minimumActionStd = o.minimumActionStd
    tunedConfig.maximumActionStd = o.maximumActionStd
    tunedConfig.finalActionStd = o.finalActionStd
    tunedConfig.actionStdAnnealStartUpdate = o.actionStdAnnealStartUpdate
    tunedConfig.actionStdAnnealEndUpdate = o.actionStdAnnealEndUpdate
    tunedConfig.initializationCheckpoint = o.initializeFrom
    tunedConfig.policyExpertInitializationCheckpoint = o.policyExpertFrom
    tunedConfig.policyExpertBranchInitializationCheckpoint =
        o.policyExpertBranchFrom
    tunedConfig.standExpertInitializationCheckpoint = o.standExpertFrom
    tunedConfig.initializationNormalizerPriorCount =
        o.initializationNormalizerPriorCount
    tunedConfig.useTaskSymmetryAugmentation = o.symmetryAugmentation
    tunedConfig.symmetryMirrorLossCoefficient =
        o.symmetryMirrorLossCoefficient
    tunedConfig.updateObservationNormalizer = o.updateObservationNormalizer
    tunedConfig.normalizeObservations = o.normalizeObservations
    tunedConfig.entropyCoefficient = o.entropy
    tunedConfig.targetKL = o.targetKL
    tunedConfig.klSchedule = o.klSchedule
    tunedConfig.checkpointInterval = o.checkpointInterval
    let trainer = VectorPPOTrainer(configuration: tunedConfig)
    try trainer.train(task: task, outputDirectory: "runs/\(taskID)/\(o.runName)",
                      resume: o.resume)

case "eval-rl":
    guard args.count > 1 else {
        fail("usage: avbd eval-rl <task> [--envs N --episodes N --seed N "
            + "--allow-task-transfer]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let taskID = args[1]
    let checkpointDirectory = o.checkpoint ?? "runs/\(taskID)/\(o.runName)"
    let checkpointMetadata = try JSONDecoder().decode(
        VectorPolicyMetadata.self,
        from: Data(contentsOf: URL(
            fileURLWithPath: "\(checkpointDirectory)/metadata.json")))
    var evaluationOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
        for: taskID,
        semanticOptions: checkpointMetadata.taskConfiguration ?? [:],
        maxEpisodeSteps: checkpointMetadata.maxEpisodeSteps,
        controlDecimation: checkpointMetadata.controlDecimation)
    // Explicit evaluation overrides are patches to the serialized task, not
    // a replacement configuration. Replacing the dictionary silently reset
    // every omitted physics/curriculum option to today's defaults, making a
    // one-parameter ablation change several unrelated variables at once.
    for (name, value) in o.taskOptions {
        evaluationOptions[name] = value
    }
    let task = try BuiltInRLTasks.registry.make(
        taskID, configuration: RLTaskConfiguration(
            numEnvironments: o.envs, seed: o.seed, autoReset: false,
            options: evaluationOptions))
    let metrics = try VectorPPOTrainer.evaluate(
        task: task, checkpointDirectory: checkpointDirectory,
        episodes: o.episodes, seed: o.seed,
        allowTaskConfigurationTransfer: o.allowTaskTransfer)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let encodedMetrics = try encoder.encode(metrics)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encodedMetrics.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: encodedMetrics, as: UTF8.self))
    } else {
        print(String(format:
            "eval %@  episodes %d  success %d (%.1f%%)  return %+.3f  length %.1f",
            taskID, metrics.episodes, metrics.successes, metrics.successRate * 100,
            metrics.meanReturn, metrics.meanEpisodeLength))
        for (name, value) in metrics.taskMetrics.sorted(by: { $0.key < $1.key }) {
            print(String(format: "  %@ %.5f", name, value))
        }
        if let acceptance = metrics.acceptance {
            print("acceptance \(acceptance.passed ? "PASS" : "FAIL")")
            for failure in acceptance.failures { print("  \(failure)") }
        }
    }
    if metrics.acceptance?.passed == false { exit(2) }

case "prepare-policy-requalification":
    guard args.count > 1 else {
        fail("usage: avbd prepare-policy-requalification <task> "
            + "--checkpoint PARENT --output CANDIDATE "
            + "--expected-checkpoint-fingerprint SHA256 "
            + "--source-commit COMMIT --inference-batch-size N "
            + "--evaluation-environments N --evaluation-seeds S1,S2,S3,S4")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let taskID = args[1]
    guard let parent = o.checkpoint,
          let output = o.output,
          let expectedFingerprint = o.expectedCheckpointFingerprint,
          let sourceCommit = o.sourceCommit,
          let inferenceBatchSize = o.inferenceBatchSize,
          let evaluationEnvironments = o.evaluationEnvironments,
          let evaluationSeeds = o.evaluationSeeds else {
        fail("prepare-policy-requalification requires every lineage and "
            + "qualification-plan option shown in --help")
    }
    let targetTask = try makeCurrentCheckpointTask(
        taskID: taskID, checkpointDirectory: parent,
        seed: evaluationSeeds[0])
    guard let criteriaProvider = targetTask as?
            any RLEvaluationCriteriaProviding else {
        fail("requalification preparation requires task-owned evaluation criteria")
    }
    let manifest = try VectorPolicyRequalification.prepare(
        targetSpec: targetTask.spec,
        parentCheckpointDirectory: parent,
        outputDirectory: output,
        expectedParentCheckpointFingerprint: expectedFingerprint,
        inferenceBatchSize: inferenceBatchSize,
        declaredSourceCommit: sourceCommit,
        qualificationPlan: .init(
            evaluationSeeds: evaluationSeeds,
            evaluationEnvironments: evaluationEnvironments),
        evaluationCriteria: criteriaProvider.evaluationCriteria)
    print("prepared zero-update \(manifest.task) revision "
        + "\(manifest.sourceTaskRevision) -> \(manifest.targetTaskRevision) "
        + "candidate \(manifest.candidateCheckpointFingerprint) at \(output)")

case "publish-policy-requalification":
    guard args.count > 1 else {
        fail("usage: avbd publish-policy-requalification <task> "
            + "--checkpoint CANDIDATE --parent-checkpoint PARENT "
            + "--qualification-directory DIR --output BUNDLE")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let taskID = args[1]
    guard let candidate = o.checkpoint,
          let parent = o.parentCheckpoint,
          let qualificationDirectory = o.qualificationDirectory,
          let output = o.output else {
        fail("publish-policy-requalification requires candidate, parent, "
            + "qualification directory, and output")
    }
    let manifestData = try Data(contentsOf: URL(
        fileURLWithPath: candidate).appendingPathComponent(
            VectorPolicyRequalification.manifestFileName))
    let prepared = try JSONDecoder().decode(
        VectorPolicyRequalificationManifest.self, from: manifestData)
    let reportPaths = prepared.qualificationPlan.evaluationSeeds.map {
        "\(qualificationDirectory)/eval-seed-\($0).json"
    }
    let targetTask = try makeCurrentCheckpointTask(
        taskID: taskID, checkpointDirectory: candidate,
        seed: prepared.qualificationPlan.evaluationSeeds[0])
    guard let criteriaProvider = targetTask as?
            any RLEvaluationCriteriaProviding else {
        fail("requalification publication requires task-owned evaluation criteria")
    }
    let manifest = try VectorPolicyRequalification.publish(
        targetSpec: targetTask.spec,
        evaluationCriteria: criteriaProvider.evaluationCriteria,
        candidateDirectory: candidate,
        parentCheckpointDirectory: parent,
        evaluationReportPaths: reportPaths,
        aggregatePath: "\(qualificationDirectory)/aggregate.json",
        outputDirectory: output)
    print("published requalified \(manifest.task) revision "
        + "\(manifest.targetTaskRevision) checkpoint "
        + "\(manifest.candidateCheckpointFingerprint) at \(output)")

case "verify-policy-requalification":
    guard args.count > 1 else {
        fail("usage: avbd verify-policy-requalification <task> "
            + "--checkpoint BUNDLE --parent-checkpoint PARENT")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let taskID = args[1]
    guard let bundle = o.checkpoint,
          let parent = o.parentCheckpoint else {
        fail("verify-policy-requalification requires bundle and parent checkpoint")
    }
    let targetTask = try makeCurrentCheckpointTask(
        taskID: taskID, checkpointDirectory: bundle)
    guard let criteriaProvider = targetTask as?
            any RLEvaluationCriteriaProviding else {
        fail("requalification verification requires task-owned evaluation criteria")
    }
    let manifest = try VectorPolicyRequalification.verify(
        targetSpec: targetTask.spec,
        evaluationCriteria: criteriaProvider.evaluationCriteria,
        bundleDirectory: bundle,
        parentCheckpointDirectory: parent)
    print("verified requalified \(manifest.task) revision "
        + "\(manifest.targetTaskRevision) checkpoint "
        + manifest.candidateCheckpointFingerprint)

case "export-policy-rl":
    guard args.count > 1 else {
        fail("usage: avbd export-policy-rl <task> --checkpoint DIR --output DIR")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    guard let checkpoint = o.checkpoint, let output = o.output else {
        fail("export-policy-rl requires --checkpoint DIR and --output DIR")
    }
    let manifest = try VectorPolicyDeploymentBundle.export(
        checkpointDirectory: checkpoint, outputDirectory: output)
    guard manifest.task == args[1] else {
        try? FileManager.default.removeItem(atPath: output)
        fail("checkpoint task \(manifest.task) does not match requested \(args[1])")
    }
    let manifestData = try JSONEncoder().encode(manifest)
    print("exported \(manifest.task) policy bundle to \(output) "
        + "(\(manifest.checkpointFingerprint), "
        + "\(String(format: "%.1f", manifest.controlFrequencyHz)) Hz, "
        + "\(manifestData.count)-byte manifest)")

case "verify-policy-rl":
    guard args.count > 1 else {
        fail("usage: avbd verify-policy-rl <task> --checkpoint BUNDLE "
            + "[--frames N --json]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    guard let bundle = o.checkpoint else {
        fail("verify-policy-rl requires --checkpoint BUNDLE")
    }
    guard o.frames > 0 else { fail("--frames must be positive") }
    let taskID = args[1]
    let runtime = try VectorPolicyDeploymentRuntime(
        bundleDirectory: bundle, expectedTask: taskID)
    let checkpointRunner = try VectorPolicyRunner(
        checkpointDirectory: bundle)
    var observation = ContiguousArray(
        repeating: Float(0), count: runtime.observationDimension)
    if taskID.hasPrefix("arachne15-")
        && runtime.observationDimension == Arachne15PolicyContract.observationDimension {
        observation[8] = 1
        observation[9] = 0.15
    }
    let deployed = try runtime.actions(for: observation)
    let replayed = try checkpointRunner.actions(for: observation)
    let maximumParityError = zip(deployed, replayed).map {
        abs($0.0 - $0.1)
    }.max() ?? .infinity
    guard maximumParityError <= 1e-7,
          deployed.allSatisfy(\.isFinite) else {
        fail("deployed inference differs from checkpoint replay")
    }
    for _ in 0..<5 { _ = try runtime.actions(for: observation) }
    var milliseconds = [Double]()
    milliseconds.reserveCapacity(o.frames)
    for _ in 0..<o.frames {
        let start = DispatchTime.now().uptimeNanoseconds
        _ = try runtime.actions(for: observation)
        let end = DispatchTime.now().uptimeNanoseconds
        milliseconds.append(Double(end - start) / 1_000_000)
    }
    milliseconds.sort()
    func percentile(_ fraction: Double) -> Double {
        milliseconds[min(Int(Double(milliseconds.count - 1) * fraction),
                         milliseconds.count - 1)]
    }
    let report: [String: Any] = [
        "task": runtime.manifest.task,
        "taskRevision": runtime.manifest.taskRevision,
        "checkpointFingerprint": runtime.checkpointFingerprint,
        "iterations": o.frames,
        "maximumParityError": maximumParityError,
        "p50Milliseconds": percentile(0.50),
        "p95Milliseconds": percentile(0.95),
        "p99Milliseconds": percentile(0.99),
        "maximumMilliseconds": milliseconds.last!,
        "controlDeadlineMilliseconds":
            runtime.controlPeriodSeconds * 1_000,
    ]
    if o.json {
        let data = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    } else {
        print("verified \(runtime.manifest.task) r\(runtime.manifest.taskRevision) "
            + "\(runtime.checkpointFingerprint)")
        print(String(format:
            "inference %d runs: p50 %.3f ms  p95 %.3f ms  p99 %.3f ms  "
                + "max %.3f ms  deadline %.1f ms  parity %.1e",
            o.frames, percentile(0.50), percentile(0.95), percentile(0.99),
            milliseconds.last!, runtime.controlPeriodSeconds * 1_000,
            maximumParityError))
    }

case "trace-rl":
    guard args.count > 1 else {
        fail("usage: avbd trace-rl <task> [--checkpoint DIR --frames N]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let taskID = args[1]
    let checkpointDirectory = o.checkpoint ?? "runs/\(taskID)/\(o.runName)"
    let runner = try VectorPolicyRunner(
        checkpointDirectory: checkpointDirectory)
    let metadata = runner.metadata
    let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
        for: taskID,
        semanticOptions: metadata.taskConfiguration ?? [:],
        maxEpisodeSteps: metadata.maxEpisodeSteps,
        controlDecimation: metadata.controlDecimation)
    let task = try BuiltInRLTasks.registry.make(
        taskID, configuration: RLTaskConfiguration(
            numEnvironments: 1, seed: o.seed, autoReset: false,
            options: replayOptions))
    if args.contains("--training-mode"),
       let trainingTask = task as? any TrainingModeConfigurable {
        trainingTask.setTrainingMode(true)
        trainingTask.setTrainingProgress(environmentSteps: Int(
            o.taskOptions["trainingEnvironmentSteps"] ?? 0))
    }
    guard metadata.compatibilityMismatches(with: task.spec).isEmpty else {
        fail("checkpoint/task mismatch for deterministic trace")
    }
    var observation = try task.reset(seed: o.seed)
    var result = RLStepBatch(spec: task.spec)
    print("step actions[7] joints[7] tip_x tip_y block_x block_y block_yaw "
        + "goal_m coverage tcp_speed block_speed block_yaw_rate robot_contact "
        + "reward done")
    for step in 0..<min(o.frames, task.spec.maxEpisodeSteps) {
        let expertGates = (task as? any PolicyExpertGateProviding)?
            .policyExpertGates(observation.policy)
        let expertActionMask = (task as? any PolicyExpertGateProviding)?
            .policyExpertActionMask
        let standExpertGates = (task as? any PolicyStandExpertGateProviding)?
            .policyStandExpertGates(observation.policy)
        let standExpertActionMask =
            (task as? any PolicyStandExpertGateProviding)?
                .policyStandExpertActionMask
        let auxiliaryExpertGates =
            (task as? any PolicyAuxiliaryExpertGateProviding)?
                .policyAuxiliaryExpertGates(observation.policy)
        let auxiliaryExpertActionMask =
            (task as? any PolicyAuxiliaryExpertGateProviding)?
                .policyAuxiliaryExpertActionMask
        let actions = try runner.actions(
            for: observation, expertGates: expertGates,
            expertActionMask: expertActionMask,
            standExpertGates: standExpertGates,
            standExpertActionMask: standExpertActionMask,
            auxiliaryExpertGates: auxiliaryExpertGates,
            auxiliaryExpertActionMask: auxiliaryExpertActionMask)
        try task.step(actions: actions, into: &result)
        if let pushT = task as? ManiSkillPushTTask {
            let state = pushT.environment.states()[0]
            let ref = pushT.environment.refs[0]
            let goal = ref.goalPosition
            let goalDistance = simd_length(
                SIMD2(state.blockPosition.x, state.blockPosition.y) - goal)
            let coverage = pushT.environment.coverage(0, state: state)
            let forward = state.blockRotation.act(F3(1, 0, 0))
            let blockYaw = atan2(forward.y, forward.x)
            let robotBodies = Set(ref.robotBodies)
            let robotContact = pushT.environment.solver
                .activeRigidContactPairs().contains { pair in
                    (pair.0 == ref.block && robotBodies.contains(pair.1))
                        || (pair.1 == ref.block && robotBodies.contains(pair.0))
                }
            let actionText = actions.values.prefix(7).map {
                String(format: "%+.5f", $0)
            }.joined(separator: ",")
            let jointText = state.jointPositions.map {
                String(format: "%+.5f", $0)
            }.joined(separator: ",")
            print(String(format:
                "%3d [%@] [%@] %+.5f %+.5f %+.5f %+.5f %+.5f "
                + "%.5f %.5f %.5f %.5f %+.5f %@ %+.5f %@",
                step + 1, actionText, jointText,
                state.tcpPosition.x, state.tcpPosition.y,
                state.blockPosition.x, state.blockPosition.y,
                blockYaw, goalDistance, coverage,
                simd_length(state.tcpLinearVelocity),
                simd_length(state.blockLinearVelocity),
                state.blockAngularVelocity.z,
                robotContact ? "true" : "false", result.rewards[0],
                result.terminated[0] || result.truncated[0]
                    ? "true" : "false"))
        } else if let arm = task as? ArmPushTTask {
            let state = arm.environment.states()[0]
            let goalDistance = simd_length(
                arm.environment.refs[0].goalPosition - state.blockPosition)
            let coverage = arm.environment.coverage(0, state: state)
            print(String(format:
                "%3d %+.5f %+.5f %+.5f %+.5f %+.5f %+.5f "
                + "%+.5f %+.5f %.5f %.5f %+.5f %@",
                step + 1, actions.values[0], actions.values[1],
                state.jointAngles[0], state.jointAngles[1],
                state.tipPosition.x, state.tipPosition.y,
                state.blockPosition.x, state.blockPosition.y,
                goalDistance, coverage, result.rewards[0],
                result.terminated[0] || result.truncated[0]
                    ? "true" : "false"))
        } else if let arachne = task as? Arachne15LocomotionTask {
            let state = arachne.environment.states()[0]
            let clearance = result.metrics[
                "state/minimum_foot_collider_clearance_m"]?[0] ?? .nan
            if args.contains("--trace-actions") {
                let formatted = actions.values.map {
                    String(format: "%+.3f", $0)
                }.joined(separator: " ")
                let velocity = state.root.rotation.conjugate.act(
                    state.root.linearVelocity)
                let angular = state.root.rotation.conjugate.act(
                    state.root.angularVelocity)
                print(String(format:
                    "actions [%@] velocity (%+.3f,%+.3f) yawRate %+.3f",
                    formatted, velocity.x, velocity.y, angular.z))
            }
            print(String(format:
                "%3d action_mean_abs %.5f root_z %.6f min_foot_clearance %.6f "
                    + "reward %+.5f done %@",
                step + 1,
                actions.values.map(abs).reduce(0, +)
                    / Float(max(actions.values.count, 1)),
                state.root.position.z, clearance, result.rewards[0],
                result.terminated[0] || result.truncated[0]
                    ? "true" : "false"))
        } else if let carry = task as? HumanoidBoxCarryTask {
            let robot = carry.environment.states()[0]
            let item = carry.environment.manipulationStates()[0]
            let forward = robot.root.rotation.act(F3(1, 0, 0))
            let rootYaw = atan2(forward.y, forward.x)
            let command = F3(
                observation.policy[9],
                observation.policy[10],
                observation.policy[11])
            let left = result.metrics["state/left_hand_contact"]?[0] ?? 0
            let right = result.metrics["state/right_hand_contact"]?[0] ?? 0
            let reach = result.metrics["state/reach_distance_m"]?[0] ?? .nan
            let distance = result.metrics["state/carry_distance_m"]?[0] ?? 0
            let phase = result.metrics["state/task_phase"]?[0] ?? .nan
            let clearance = result.metrics["state/box_clearance_m"]?[0] ?? .nan
            let missed = result.metrics[
                "state/missed_bilateral_contact_steps"]?[0] ?? 0
            let ground = result.metrics["state/box_ground_contact"]?[0] ?? 0
            let pedestal = result.metrics[
                "state/box_pedestal_contact"]?[0] ?? 0
            let destination = result.metrics[
                "state/box_destination_contact"]?[0] ?? 0
            let placement = result.metrics[
                "state/placement_distance_m"]?[0] ?? .nan
            let released = result.metrics["state/released"]?[0] ?? 0
            let stableUnsupported = result.metrics[
                "state/stable_unsupported_steps"]?[0] ?? 0
            let loadedRoot = result.metrics[
                "state/loaded_root_displacement_m"]?[0] ?? 0
            let loadedSteps = result.metrics[
                "state/loaded_alternating_steps"]?[0] ?? 0
            let leftFoot = result.metrics["state/left_foot_contact"]?[0] ?? 0
            let rightFoot = result.metrics["state/right_foot_contact"]?[0] ?? 0
            let handoff = result.metrics[
                "state/carry_handoff_progress"]?[0] ?? 0
            let manipulationHandoff = result.metrics[
                "state/manipulation_handoff_progress"]?[0] ?? 0
            let commandRamp = result.metrics[
                "state/carry_command_progress"]?[0] ?? 0
            if args.contains("--trace-actions") {
                let formatted = actions.values.map {
                    String(format: "%+.5f", $0)
                }.joined(separator: ",")
                print("actions [\(formatted)]")
            }
            print(String(format:
                "%3d root (%+.3f,%+.3f,%.3f) yaw %+.3f "
                    + "cmd (%+.3f,%+.3f,%+.3f) box (%+.3f,%+.3f,%.3f) "
                    + "hands_z (%.3f,%.3f) contact %.0f/%.0f reach %.3f "
                    + "carry %.3f root_carry %.3f steps %.0f feet %.0f/%.0f "
                    + "phase %.0f clearance %.3f missed %.0f "
                    + "support %.0f/%.0f/%.0f goal %.3f released %.0f hold %.0f "
                    + "manip %.2f handoff %.2f command %.2f "
                    + "reward %+.4f done %@",
                step + 1, robot.root.position.x, robot.root.position.y,
                robot.root.position.z, rootYaw,
                command.x, command.y, command.z,
                item.object.position.x, item.object.position.y,
                item.object.position.z,
                item.leftHand.position.z, item.rightHand.position.z,
                left, right, reach, distance, loadedRoot, loadedSteps,
                leftFoot, rightFoot,
                phase, clearance, missed,
                ground, pedestal, destination, placement, released,
                stableUnsupported,
                manipulationHandoff, handoff, commandRamp, result.rewards[0],
                result.terminated[0] || result.truncated[0]
                    ? "true" : "false"))
        } else {
            print(String(format: "%3d action_mean_abs %.5f reward %+.5f done %@",
                         step + 1,
                         actions.values.map(abs).reduce(0, +)
                            / Float(max(actions.values.count, 1)),
                         result.rewards[0],
                result.terminated[0] || result.truncated[0]
                    ? "true" : "false"))
        }
        observation = result.observations
        if result.terminated[0] || result.truncated[0] { break }
    }

case "distill-h1-box-lift":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let checkpoint = o.checkpoint, o.runName != "ppo" else {
        fail("usage: avbd distill-h1-box-lift --checkpoint DIR "
            + "--run PROBE_REPORT_JSON --output DIR")
    }
    let output = o.output ?? "runs/humanoid-box-carry-v0/carry-distilled"
    let report = try HumanoidBoxCarryDistillation.run(
        checkpointDirectory: checkpoint,
        probeReportPath: o.runName,
        outputDirectory: output,
        configuration: .init(
            collectionEnvironments: o.envs,
            contactDwellSteps: Int(
                o.taskOptions["contactDwellSteps"] ?? 12),
            probeSteps: o.frames,
            epochs: o.updates,
            learningRate: o.lr,
            maximumGradientNorm: o.maxGradientNorm,
            aggregationRounds: Int(
                o.taskOptions["aggregationRounds"] ?? 1),
            initialTeacherMix:
                o.taskOptions["initialTeacherMix"] ?? 0.75,
            finalTeacherMix:
                o.taskOptions["finalTeacherMix"] ?? 0,
            rolloutArmNoiseStandardDeviation:
                o.taskOptions["rolloutArmNoiseStandardDeviation"] ?? 0.01,
            stateAlignedTeacherQueries:
                o.taskOptions["stateAlignedTeacherQueries"] == 1,
            physicalStateWeighting:
                o.taskOptions["physicalStateWeighting"] == 1,
            armActionWeight:
                o.taskOptions["armActionWeight"] ?? 1,
            probeSeed: o.seed))
    print(String(format:
        "distilled %d -> %d rows/%d rounds, clearance %.4fm, carry %.3fm/%dfr, "
            + "action MSE %.6g -> %.6g, snapshots %d",
        report.teacherRows, report.aggregatedRows, report.aggregationRounds,
        report.teacherMaximumClearanceMeters,
        report.teacherMaximumCarryDistanceMeters,
        report.teacherMaximumStableUnsupportedSteps,
        report.initialMeanSquaredActionError,
        report.finalMeanSquaredActionError, report.snapshots.count))

case "distill-h1-box-flow":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let checkpoint = o.checkpoint, o.runName != "ppo",
          let output = o.output else {
        fail("usage: avbd distill-h1-box-flow --checkpoint DIR "
            + "--run FLOW_REPORT_JSON --output DIR [--updates EPOCHS]")
    }
    let report = try HumanoidBoxFlowDistillation.run(
        checkpointDirectory: checkpoint,
        flowReportPath: o.runName,
        outputDirectory: output,
        configuration: .init(
            collectionEnvironments: o.envs,
            contactDwellSteps: Int(
                o.taskOptions["contactDwellSteps"] ?? 8),
            epochs: args.contains("--updates") ? o.updates : 200,
            learningRate: o.lr,
            maximumGradientNorm: o.maxGradientNorm,
            targetRowWeight:
                o.taskOptions["targetRowWeight"] ?? 4,
            policySourceRowWeight:
                o.taskOptions["policySourceRowWeight"] ?? 4,
            aggregationRounds: Int(
                o.taskOptions["aggregationRounds"] ?? 4),
            initialTeacherMix:
                o.taskOptions["initialTeacherMix"] ?? 0.75,
            finalTeacherMix:
                o.taskOptions["finalTeacherMix"] ?? 0,
            rolloutActionNoiseStandardDeviation:
                o.taskOptions["rolloutActionNoiseStandardDeviation"]
                    ?? 0.005,
            stateAlignmentLookahead: Int(
                o.taskOptions["stateAlignmentLookahead"] ?? 12),
            validationActionNoiseStandardDeviation:
                o.taskOptions[
                    "validationActionNoiseStandardDeviation"] ?? 0.001,
            validationNoiseIncludesSource:
                o.taskOptions["validationNoiseIncludesSource"] == 1,
            robustReplayCount: Int(
                o.taskOptions["robustReplayCount"] ?? 32),
            trainOutputHeadsOnly:
                o.taskOptions["trainOutputHeadsOnly"] != 0,
            seed: o.seed))
    print(String(format:
        "flow distillation %d -> %d rows/%d rounds action MSE %.6g -> %.6g, teacher %.3fm learner %.3fm deterministic %.0f%% robust %.0f%% best@%d %@",
        report.teacherRows, report.aggregatedRows,
        report.aggregationRounds, report.initialActionMSE,
        report.finalActionMSE, report.teacherCarryDistanceMeters,
        report.learnerMaximumStableCarryDistanceMeters,
        100 * report.learnerDeterministicFinalSuccessFraction,
        100 * report.learnerRobustFinalSuccessFraction,
        report.bestEpoch,
        report.distillationGatePassed ? "PASS" : "FAIL"))

case "interpolate-h1-box-flow":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let base = o.checkpoint,
          let candidate = o.initializeFrom,
          o.runName != "ppo", let output = o.output else {
        fail("usage: avbd interpolate-h1-box-flow --checkpoint BASE_DIR "
            + "--initialize-from CANDIDATE_DIR --run FLOW_REPORT_JSON "
            + "--output DIR")
    }
    let report = try HumanoidBoxFlowDistillation.evaluateResidualScales(
        baseCheckpointDirectory: base,
        candidateCheckpointDirectory: candidate,
        flowReportPath: o.runName,
        outputDirectory: output,
        configuration: .init(
            collectionEnvironments: o.envs,
            validationActionNoiseStandardDeviation:
                o.taskOptions[
                    "validationActionNoiseStandardDeviation"] ?? 0.001,
            validationNoiseIncludesSource:
                o.taskOptions["validationNoiseIncludesSource"] == 1,
            robustReplayCount: Int(
                o.taskOptions["robustReplayCount"] ?? 32),
            seed: o.seed))
    print(String(format:
        "flow residual scale %.6f carry %.3fm deterministic %.0f%% robust %.0f%% %@",
        report.selectedScale,
        report.selectedMaximumStableCarryDistanceMeters,
        100 * report.selectedDeterministicSuccessFraction,
        100 * report.selectedRobustSuccessFraction,
        report.gatePassed ? "PASS" : "FAIL"))

case "train-h1-box-action-chunk":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let checkpoint = o.checkpoint, o.runName != "ppo",
          let output = o.output else {
        fail("usage: avbd train-h1-box-action-chunk --checkpoint DIR "
            + "--run FLOW_REPORT_JSON --output DIR [--updates EPOCHS "
            + "--horizon ACTIONS]")
    }
    let report = try HumanoidBoxFlowDistillation.trainActionChunk(
        checkpointDirectory: checkpoint,
        flowReportPath: o.runName,
        outputDirectory: output,
        configuration: .init(
            horizon: o.horizon,
            hiddenDimension: o.hidden,
            epochs: args.contains("--updates") ? o.updates : 2_000,
            learningRate: o.lr,
            maximumGradientNorm: o.maxGradientNorm,
            maximumResidualAction:
                o.taskOptions["maximumResidualAction"] ?? 2,
            contactDwellSteps: Int(
                o.taskOptions["contactDwellSteps"] ?? 8),
            trainingReplayCount: Int(
                o.taskOptions["trainingReplayCount"] ?? 1),
            trainingActionNoiseStandardDeviation:
                o.taskOptions[
                    "trainingActionNoiseStandardDeviation"] ?? 0,
            exactReplayRowWeight:
                o.taskOptions["exactReplayRowWeight"] ?? 1,
            exactFineTuneEpochs: Int(
                o.taskOptions["exactFineTuneEpochs"] ?? 0),
            exactFineTuneLearningRate:
                o.taskOptions["exactFineTuneLearningRate"] ?? 1e-5,
            exactActionBiasWeight:
                o.taskOptions["exactActionBiasWeight"] ?? 0,
            robustReplayCount: Int(
                o.taskOptions["robustReplayCount"] ?? 32),
            validationActionNoiseStandardDeviation:
                o.taskOptions[
                    "validationActionNoiseStandardDeviation"] ?? 0.001,
            seed: o.seed))
    print(String(format:
        "flow action chunk h%d/%d rows residual MSE %.6g -> tube %.6g -> exact %.6g, max %.6g bias %.6g, zero %.3fm/%.0f%% teacher %.3fm/%.0f%% teacher-robust %.3fm/%.0f%% learned %.3fm/%.0f%% robust %.3fm/%.0f%% %@",
        report.horizon, report.trainingRows,
        report.initialMeanSquaredResidualError,
        report.tubeMeanSquaredResidualError,
        report.finalMeanSquaredResidualError,
        report.maximumExactFirstActionResidualError,
        report.maximumAbsoluteExactActionBias,
        report.zeroResidualMaximumStableCarryDistanceMeters,
        100 * report.zeroResidualFinalSuccessFraction,
        report.teacherMaximumStableCarryDistanceMeters,
        100 * report.teacherFinalSuccessFraction,
        report.teacherRobustMaximumStableCarryDistanceMeters,
        100 * report.teacherRobustFinalSuccessFraction,
        report.learnedMaximumStableCarryDistanceMeters,
        100 * report.learnedFinalSuccessFraction,
        report.learnedRobustMaximumStableCarryDistanceMeters,
        100 * report.learnedRobustFinalSuccessFraction,
        report.gatePassed ? "PASS" : "FAIL"))

case "eval-h1-box-action-chunk":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let checkpoint = o.checkpoint,
          let actionChunk = o.initializeFrom,
          o.runName != "ppo" else {
        fail("usage: avbd eval-h1-box-action-chunk --checkpoint SOURCE_DIR "
            + "--initialize-from ACTION_CHUNK_DIR --run FLOW_REPORT_JSON "
            + "[--envs REPLAYS --seed SEED --output REPORT_JSON]")
    }
    let report = try HumanoidBoxFlowDistillation.evaluateActionChunk(
        checkpointDirectory: checkpoint,
        flowReportPath: o.runName,
        actionChunkCheckpointDirectory: actionChunk,
        replayCount: o.envs,
        actionNoiseStandardDeviation:
            o.taskOptions[
                "validationActionNoiseStandardDeviation"] ?? 0.001,
        seed: o.seed)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: data, as: UTF8.self))
    } else {
        print(String(format:
            "flow action chunk eval seed %llu h%d noise %.4f teacher %.3fm destination %+0.3fm/%.0f%% learned %.3fm destination %+0.3fm/%.0f%% %@ fingerprint %@",
            report.evaluationSeed, report.targetControlSteps,
            report.actionNoiseStandardDeviation,
            report.teacherMaximumStableCarryDistanceMeters,
            report.teacherMaximumStableDestinationProgressMeters ?? 0,
            100 * report.teacherSuccessFraction,
            report.learnedMaximumStableCarryDistanceMeters,
            report.learnedMaximumStableDestinationProgressMeters ?? 0,
            100 * report.learnedSuccessFraction,
            report.gatePassed ? "PASS" : "FAIL",
            String(report.actionChunkFingerprint.prefix(12))))
    }
    if !report.gatePassed { exit(2) }

case "eval-h1-box-flow":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let checkpoint = o.checkpoint, o.runName != "ppo" else {
        fail("usage: avbd eval-h1-box-flow --checkpoint DIR "
            + "--run FLOW_REPORT_JSON [--envs REPLAYS]")
    }
    let configuration = HumanoidBoxFlowDistillationConfiguration(
        collectionEnvironments: 1,
        contactDwellSteps: Int(
            o.taskOptions["contactDwellSteps"] ?? 8),
        epochs: 1,
        aggregationRounds: 1,
        validationActionNoiseStandardDeviation:
            o.taskOptions[
                "validationActionNoiseStandardDeviation"] ?? 0.001,
        validationNoiseIncludesSource:
            o.taskOptions["validationNoiseIncludesSource"] == 1,
        robustReplayCount: o.envs,
        seed: o.seed)
    let report = try HumanoidBoxFlowDistillation.evaluate(
        checkpointDirectory: checkpoint,
        flowReportPath: o.runName,
        configuration: configuration)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: data, as: UTF8.self))
    } else {
        print(String(format:
            "H1 box flow eval required %.3fm deterministic %.3fm/%.0f%% %@ noise %.4f robust %.3fm/%.0f%% %@",
            report.requiredCarryDistanceMeters,
            report.deterministicMaximumStableCarryDistanceMeters,
            100 * report.deterministicFinalSuccessFraction,
            report.deterministicGatePassed ? "PASS" : "FAIL",
            report.actionNoiseStandardDeviation,
            report.robustMaximumStableCarryDistanceMeters,
            100 * report.robustFinalSuccessFraction,
            report.robustGatePassed ? "PASS" : "FAIL"))
    }

case "experiment-h1-box-flow":
    guard args.count > 1, !args[1].hasPrefix("--") else {
        fail("usage: avbd experiment-h1-box-flow <target-probe.json> "
            + "[proposal-probe.json] --checkpoint DIR [options]")
    }
    let targetData = try Data(contentsOf: URL(fileURLWithPath: args[1]))
    let hasProposal = args.count > 2 && !args[2].hasPrefix("--")
    let o = parseOptions(Array(args.dropFirst(hasProposal ? 3 : 2)))
    let requestedArchiveSelectors = [
        "useLastTargetGeneratingActionAsTargetSeed",
        "useBestStableSwingAsTargetSeed",
        "useBestSwingFrontierAsTargetSeed",
        "useBestFeasibilityFrontierAsTargetSeed",
        "useBestTrajectoryAsTargetSeed",
    ].filter { o.taskOptions[$0] == 1 }
    guard requestedArchiveSelectors.count <= 1 else {
        fail("select exactly one target-seed archive")
    }
    var targetTrajectory: [Float]
    if let requestedIndex = o.taskOptions[
            "targetGeneratingActionIndex"].map({ Int($0) }) {
        guard let trajectory = humanoidBoxTargetGeneratingAction(
            from: targetData, index: requestedIndex) else {
            fail("target-generating action index is absent or invalid")
        }
        targetTrajectory = trajectory
    } else if requestedArchiveSelectors.first
                == "useLastTargetGeneratingActionAsTargetSeed" {
        guard let trajectory = humanoidBoxLastTargetGeneratingAction(
            from: targetData) else {
            fail("last target-generating action is absent or invalid")
        }
        targetTrajectory = trajectory
    } else if requestedArchiveSelectors.first
                == "useBestStableSwingAsTargetSeed" {
        guard let trajectory = humanoidBoxNamedTrajectory(
            from: targetData, key: "bestStableSwingParameters") else {
            fail("best stable-swing archive is absent or invalid")
        }
        targetTrajectory = trajectory
    } else if requestedArchiveSelectors.first
                == "useBestSwingFrontierAsTargetSeed" {
        guard let trajectory = humanoidBoxNamedTrajectory(
            from: targetData, key: "bestSwingFrontierParameters") else {
            fail("best swing-frontier archive is absent or invalid")
        }
        targetTrajectory = trajectory
    } else if requestedArchiveSelectors.first
                == "useBestFeasibilityFrontierAsTargetSeed" {
        guard let trajectory = humanoidBoxNamedTrajectory(
            from: targetData,
            key: "bestFeasibilityFrontierParameters") else {
            fail("best feasibility-frontier archive is absent or invalid")
        }
        targetTrajectory = trajectory
    } else if requestedArchiveSelectors.first
                == "useBestTrajectoryAsTargetSeed" {
        guard let trajectory = humanoidBoxTrajectory(from: targetData) else {
            fail("best trajectory archive is absent or invalid")
        }
        targetTrajectory = trajectory
    } else if let probe = try? JSONDecoder().decode(
        HumanoidBoxCarryActuationProbeReport.self, from: targetData) {
        targetTrajectory = probe.bestArmTarget
    } else if let trajectory = humanoidBoxTargetGeneratingTrajectory(
        from: targetData) {
        targetTrajectory = trajectory
    } else {
        fail("target must be a box actuation-probe or discovered flow report")
    }
    let targetSourceSchema = humanoidBoxFlowTrajectorySchema(from: targetData)
    guard let targetArmParameterCountFromArtifact = targetSourceSchema?
            .armParameterCount(totalParameterCount: targetTrajectory.count) else {
        fail("target artifact has an invalid explicit trajectory schema")
    }
    var archivedTargetTrajectorySequence: [[Float]]?
    var archivedTargetTrajectorySequencePhaseSteps: [Int]?
    var archivedTargetPredictedRecoveryPathSafe: Bool?
    if o.taskOptions["useArchivedTargetGeneratingSequence"] == 1 {
        guard let object = try JSONSerialization.jsonObject(
                with: targetData) as? [String: Any],
              let encodedSequence = object[
                "targetGeneratingTrajectorySequence"] as? [[NSNumber]],
              !encodedSequence.isEmpty else {
            fail("archived target sequence is absent or invalid")
        }
        archivedTargetTrajectorySequence = encodedSequence.map {
            $0.map(\.floatValue)
        }
        archivedTargetTrajectorySequencePhaseSteps = (object[
            "targetGeneratingTrajectorySequencePhaseSteps"]
            as? [NSNumber])?.map(\.intValue)
        let archivedDenominator = ((object["recedingHorizonSteps"]
            ?? object["targetGenerationSteps"])
            as? NSNumber)?.intValue
        guard archivedTargetTrajectorySequence?.allSatisfy({
                  $0.count == targetTrajectory.count
                    && $0.allSatisfy(\.isFinite)
              }) == true,
              (archivedTargetTrajectorySequencePhaseSteps.map { phaseSteps in
                  phaseSteps.count == archivedTargetTrajectorySequence?.count
                    && archivedDenominator.map { denominator in
                        denominator > 0 && phaseSteps.allSatisfy {
                            $0 >= 0 && $0 < denominator
                        }
                    } == true
              } ?? true) else {
            fail("archived target sequence does not match its declared schema")
        }
        archivedTargetPredictedRecoveryPathSafe = object[
            "targetPredictedRecoveryPathSafe"] as? Bool
    }
    let proposalData: Data?
    var proposalTrajectory: [Float]?
    if hasProposal {
        let data = try Data(contentsOf: URL(fileURLWithPath: args[2]))
        proposalData = data
        if let trajectory = humanoidBoxTrajectory(from: data) {
            proposalTrajectory = trajectory
        } else {
            fail("proposal must be a box actuation-probe or physical-flow report")
        }
    } else {
        proposalData = nil
        proposalTrajectory = nil
    }
    guard let checkpoint = o.checkpoint else {
        fail("experiment-h1-box-flow requires --checkpoint DIR")
    }
    let legBlendKnotCount = Int(
        o.taskOptions["legBlendKnotCount"] ?? 0)
    let legResidualKnotCount = Int(
        o.taskOptions["legResidualKnotCount"] ?? 0)
    let torsoResidualKnotCount = Int(
        o.taskOptions["torsoResidualKnotCount"] ?? 0)
    let maximumLegResidualAction =
        o.taskOptions["maximumLegResidualAction"] ?? 0.25
    let maximumTorsoResidualAction =
        o.taskOptions["maximumTorsoResidualAction"] ?? 0.25
    let armAsymmetryKnotCount = Int(
        o.taskOptions["armAsymmetryKnotCount"] ?? 0)
    let maximumArmAsymmetryAction =
        o.taskOptions["maximumArmAsymmetryAction"] ?? 0.25
    let targetDiscoveryPopulationSize = Int(
        o.taskOptions["targetDiscoveryPopulationSize"] ?? 0)
    if let sourceSchema = targetSourceSchema,
       sourceSchema.blendKnots == legBlendKnotCount,
       (sourceSchema.residualKnots != legResidualKnotCount
        || sourceSchema.torsoResidualKnots != torsoResidualKnotCount
        || sourceSchema.residualMaximum != maximumLegResidualAction
        || sourceSchema.torsoResidualMaximum
            != maximumTorsoResidualAction
        || sourceSchema.armAsymmetryKnots != armAsymmetryKnotCount
        || sourceSchema.armAsymmetryMaximum
            != maximumArmAsymmetryAction) {
        guard let armCount = sourceSchema.armParameterCount(
            totalParameterCount: targetTrajectory.count) else {
            fail("target flow trajectory has an invalid source schema")
        }
        targetTrajectory = HumanoidBoxPhysicalFlowExperiment
            .resampledLegResidualTrajectory(
                targetTrajectory,
                armParameterCount: armCount,
                blendKnotCount: sourceSchema.blendKnots,
                sourceResidualKnotCount: sourceSchema.residualKnots,
                targetResidualKnotCount: legResidualKnotCount,
                sourceResidualMaximumAction:
                    sourceSchema.residualMaximum,
                targetResidualMaximumAction:
                    maximumLegResidualAction,
                sourceTorsoResidualKnotCount:
                    sourceSchema.torsoResidualKnots,
                targetTorsoResidualKnotCount:
                    torsoResidualKnotCount,
                sourceTorsoResidualMaximumAction:
                    sourceSchema.torsoResidualMaximum,
                targetTorsoResidualMaximumAction:
                    maximumTorsoResidualAction,
                sourceArmAsymmetryKnotCount:
                    sourceSchema.armAsymmetryKnots,
                targetArmAsymmetryKnotCount:
                    armAsymmetryKnotCount,
                sourceArmAsymmetryMaximumAction:
                    sourceSchema.armAsymmetryMaximum,
                targetArmAsymmetryMaximumAction:
                    maximumArmAsymmetryAction)
    }
    if let proposalData,
       let sourceSchema = humanoidBoxFlowTrajectorySchema(from: proposalData),
       sourceSchema.blendKnots == legBlendKnotCount,
       (sourceSchema.residualKnots != legResidualKnotCount
        || sourceSchema.torsoResidualKnots != torsoResidualKnotCount
        || sourceSchema.residualMaximum != maximumLegResidualAction
        || sourceSchema.torsoResidualMaximum
            != maximumTorsoResidualAction
        || sourceSchema.armAsymmetryKnots != armAsymmetryKnotCount
        || sourceSchema.armAsymmetryMaximum
            != maximumArmAsymmetryAction),
       let trajectory = proposalTrajectory {
        guard let armCount = sourceSchema.armParameterCount(
            totalParameterCount: trajectory.count) else {
            fail("proposal flow trajectory has an invalid source schema")
        }
        proposalTrajectory = HumanoidBoxPhysicalFlowExperiment
            .resampledLegResidualTrajectory(
                trajectory,
                armParameterCount: armCount,
                blendKnotCount: sourceSchema.blendKnots,
                sourceResidualKnotCount: sourceSchema.residualKnots,
                targetResidualKnotCount: legResidualKnotCount,
                sourceResidualMaximumAction:
                    sourceSchema.residualMaximum,
                targetResidualMaximumAction:
                    maximumLegResidualAction,
                sourceTorsoResidualKnotCount:
                    sourceSchema.torsoResidualKnots,
                targetTorsoResidualKnotCount:
                    torsoResidualKnotCount,
                sourceTorsoResidualMaximumAction:
                    sourceSchema.torsoResidualMaximum,
                targetTorsoResidualMaximumAction:
                    maximumTorsoResidualAction,
                sourceArmAsymmetryKnotCount:
                    sourceSchema.armAsymmetryKnots,
                targetArmAsymmetryKnotCount:
                    armAsymmetryKnotCount,
                sourceArmAsymmetryMaximumAction:
                    sourceSchema.armAsymmetryMaximum,
                targetArmAsymmetryMaximumAction:
                    maximumArmAsymmetryAction)
    }
    let legResidualParameterCount = 10 * legResidualKnotCount
    let torsoResidualParameterCount = torsoResidualKnotCount
    let armAsymmetryParameterCount = 4 * armAsymmetryKnotCount
    // The artifact declares every structured action head. Derive the arm
    // count from that schema before any resampling instead of guessing from
    // divisibility: structured trajectories can also have a total length
    // divisible by four.
    let targetArmParameterCount = targetArmParameterCountFromArtifact
    let knotCount = targetArmParameterCount / 4
    let acceptedTrajectoryCounts = Set([
        targetArmParameterCount,
        targetArmParameterCount + legBlendKnotCount,
        targetArmParameterCount + legBlendKnotCount
            + legResidualParameterCount,
        targetArmParameterCount + legBlendKnotCount
            + legResidualParameterCount + torsoResidualParameterCount,
        targetArmParameterCount + legBlendKnotCount
            + legResidualParameterCount + torsoResidualParameterCount
            + armAsymmetryParameterCount,
    ])
    if let proposalTrajectory,
       !acceptedTrajectoryCounts.contains(proposalTrajectory.count) {
        fail("proposal and target probe knot schemas differ")
    }
    var sourceStages: [HumanoidBoxPhysicalFlowStage]
    var archivedSourceWarmupActions: [[Float]]?
    if o.runName != "ppo" {
        let data = try Data(contentsOf: URL(fileURLWithPath: o.runName))
        if let object = try JSONSerialization.jsonObject(with: data)
                as? [String: Any] {
            archivedSourceWarmupActions = (object[
                "sourceWarmupAppliedActions"] as? [[NSNumber]])?.map {
                    $0.map(\.floatValue)
                }
            guard archivedSourceWarmupActions.map({ actions in
                !actions.isEmpty && actions.allSatisfy {
                    $0.count == HumanoidBoxCarryTask.actionDimension
                        && $0.allSatisfy(\.isFinite)
                }
            }) ?? true else {
                fail("source warm-up controls are incomplete or invalid")
            }
        } else {
            archivedSourceWarmupActions = nil
        }
        let encodedSchema = humanoidBoxFlowTrajectorySchema(from: data)
        let useTargetGeneratingSource =
            o.taskOptions["useTargetGeneratingTrajectoryAsSource"] == 1
        let sourceStagesOnly =
            o.taskOptions["sourceStagesOnly"] == 1
        if let stages = humanoidBoxFlowSourceStages(
            from: data,
            useTargetGeneratingTrajectory:
                useTargetGeneratingSource,
            includeOwnStage: !sourceStagesOnly) {
            sourceStages = stages
        } else if useTargetGeneratingSource {
            fail("--run target-generating source is not a reusable "
                + "recovery-safe frontier")
        } else if let trajectory = humanoidBoxTrajectory(from: data) {
            // Probe artifacts predate explicit flow lineage. They remain
            // usable when the caller supplies the replay duration.
            sourceStages = [.init(
                trajectory: trajectory,
                controlSteps: Int(
                    o.taskOptions["sourceTrajectorySteps"] ?? 16))]
        } else {
            fail("--run source must be a box physical-flow or probe report")
        }
        if let requestedSteps = o.taskOptions[
                "sourceOwnStageExecutionSteps"].map({ Int($0) }) {
            guard !sourceStagesOnly,
                  let ownStage = sourceStages.last,
                  let prefix = ownStage.prefix(
                    controlSteps: requestedSteps) else {
                fail("sourceOwnStageExecutionSteps must select a positive "
                    + "prefix of the appended source stage")
            }
            sourceStages[sourceStages.count - 1] = prefix
        }
        if let encodedSchema,
           encodedSchema.blendKnots == legBlendKnotCount,
           (encodedSchema.residualKnots != legResidualKnotCount
            || encodedSchema.torsoResidualKnots
                != torsoResidualKnotCount
            || encodedSchema.residualMaximum
                != maximumLegResidualAction
            || encodedSchema.torsoResidualMaximum
                != maximumTorsoResidualAction
            || encodedSchema.armAsymmetryKnots
                != armAsymmetryKnotCount
            || encodedSchema.armAsymmetryMaximum
                != maximumArmAsymmetryAction) {
            sourceStages = sourceStages.map { stage in
                guard stage.policyOnly != true else { return stage }
                let reference = stage.trajectorySequence?.first
                    ?? stage.trajectory
                guard let armCount = encodedSchema.armParameterCount(
                    totalParameterCount: reference.count) else {
                    fail("source flow trajectory has an invalid source schema")
                }
                func converted(_ trajectory: [Float]) -> [Float] {
                    HumanoidBoxPhysicalFlowExperiment
                        .resampledLegResidualTrajectory(
                            trajectory,
                            armParameterCount: armCount,
                            blendKnotCount: encodedSchema.blendKnots,
                            sourceResidualKnotCount:
                                encodedSchema.residualKnots,
                            targetResidualKnotCount: legResidualKnotCount,
                            sourceResidualMaximumAction:
                                encodedSchema.residualMaximum,
                            targetResidualMaximumAction:
                                maximumLegResidualAction,
                            sourceTorsoResidualKnotCount:
                                encodedSchema.torsoResidualKnots,
                            targetTorsoResidualKnotCount:
                                torsoResidualKnotCount,
                            sourceTorsoResidualMaximumAction:
                                encodedSchema.torsoResidualMaximum,
                            targetTorsoResidualMaximumAction:
                                maximumTorsoResidualAction,
                            sourceArmAsymmetryKnotCount:
                                encodedSchema.armAsymmetryKnots,
                            targetArmAsymmetryKnotCount:
                                armAsymmetryKnotCount,
                            sourceArmAsymmetryMaximumAction:
                                encodedSchema.armAsymmetryMaximum,
                            targetArmAsymmetryMaximumAction:
                                maximumArmAsymmetryAction)
                }
                return HumanoidBoxPhysicalFlowStage(
                    trajectory: stage.trajectorySequence == nil
                        ? converted(stage.trajectory) : [],
                    controlSteps: stage.controlSteps,
                    trajectoryDurationSteps:
                        stage.trajectoryDurationSteps,
                    trajectorySequence:
                        stage.trajectorySequence?.map(converted),
                    trajectorySequencePhaseSteps:
                        stage.trajectorySequencePhaseSteps,
                    trajectorySequenceStepDenominator:
                        stage.trajectorySequenceStepDenominator,
                    forwardOnlyBaseCommand:
                        stage.forwardOnlyBaseCommand == true,
                    holonomicBaseCommand:
                        stage.holonomicBaseCommand == true,
                    locomotionCheckpointDirectory:
                        stage.locomotionCheckpointDirectory,
                    locomotionCommandSpeed:
                        stage.locomotionCommandSpeed,
                    policyOnly: stage.policyOnly ?? false,
                    appliedNormalizedActions:
                        stage.appliedNormalizedActions,
                    canonicalizeReplicasBeforeExecution:
                        stage.canonicalizeReplicasBeforeExecution == true,
                    continueFromPreviousTrajectoryTerminal:
                        stage.continueFromPreviousTrajectoryTerminal == true,
                    graspAnchorFeedbackBlend:
                        stage.graspAnchorFeedbackBlend,
                    graspAnchorFeedbackVelocityHorizonSeconds:
                        stage.graspAnchorFeedbackVelocityHorizonSeconds,
                    graspAnchorFeedbackMaximumActionCorrection:
                        stage.graspAnchorFeedbackMaximumActionCorrection,
                    graspAnchorFeedbackInwardPreloadMeters:
                        stage.graspAnchorFeedbackInwardPreloadMeters,
                    leftGraspAnchorBoxLocalMeters:
                        stage.leftGraspAnchorBoxLocalMeters,
                    rightGraspAnchorBoxLocalMeters:
                        stage.rightGraspAnchorBoxLocalMeters,
                    graspAnchorBoxHeightMeters:
                        stage.graspAnchorBoxHeightMeters,
                    minimumCarryDistanceMeters:
                        stage.minimumCarryDistanceMeters,
                    minimumDestinationProgressMeters:
                        stage.minimumDestinationProgressMeters,
                    minimumRootDestinationProgressMeters:
                        stage.minimumRootDestinationProgressMeters,
                    minimumTouchdowns:
                        stage.minimumTouchdowns,
                    minimumAlternatingSteps:
                        stage.minimumAlternatingSteps,
                    minimumSwingFootLiftMeters:
                        stage.minimumSwingFootLiftMeters,
                    minimumFootAirTimeSeconds:
                        stage.minimumFootAirTimeSeconds,
                    minimumFootUnloadingFraction:
                        stage.minimumFootUnloadingFraction,
                    minimumTerminalFootUnloadingFraction:
                        stage.minimumTerminalFootUnloadingFraction,
                    minimumClearanceMeters:
                        stage.minimumClearanceMeters,
                    maximumPathDownwardBoxVelocityMPS:
                        stage.maximumPathDownwardBoxVelocityMPS,
                    minimumGraspQuality:
                        stage.minimumGraspQuality,
                    certificationDwellSteps:
                        stage.certificationDwellSteps)
            }
        }
        if sourceStages.contains(where: {
            $0.policyOnly != true
                && !($0.trajectorySequence?.allSatisfy {
                    acceptedTrajectoryCounts.contains($0.count)
                } ?? acceptedTrajectoryCounts.contains($0.trajectory.count))
        }) {
            fail("source and target trajectory knot schemas differ")
        }
    } else {
        sourceStages = []
        archivedSourceWarmupActions = nil
    }
    if let policySourceSteps = o.taskOptions["policySourceSteps"],
       policySourceSteps > 0 {
        sourceStages.append(.init(
            trajectory: [], controlSteps: Int(policySourceSteps),
            policyOnly: true))
    }
    let configuration = HumanoidBoxPhysicalFlowConfiguration(
        populationSize: o.envs,
        proposalProbeSize: Int(
            o.taskOptions["proposalProbeSize"] ?? 32),
        generations: args.contains("--updates") ? o.updates : 6,
        maximumWarmupSteps: Int(
            o.taskOptions["maximumWarmupSteps"] ?? 600),
        contactDwellSteps: Int(
            o.taskOptions["contactDwellSteps"] ?? 8),
        targetGenerationSteps: args.contains("--frames") ? o.frames : 400,
        targetExecutionSteps: o.taskOptions["targetExecutionSteps"].map {
            Int($0)
        },
        targetSelectionStep: o.taskOptions["targetSelectionStep"].map {
            Int($0)
        } ?? (targetDiscoveryPopulationSize == 0
            ? humanoidBoxFlowSelectedTargetStep(from: targetData) : nil),
        candidateTrajectoryDurationSteps:
            o.taskOptions["candidateTrajectoryDurationSteps"].map {
                Int($0)
            },
        targetDiscoveryPopulationSize: targetDiscoveryPopulationSize,
        targetDiscoveryGenerations: Int(
            o.taskOptions["targetDiscoveryGenerations"] ?? 0),
        targetDiscoveryInitialStandardDeviation:
            o.taskOptions["targetDiscoveryInitialStandardDeviation"] ?? 0.25,
        targetDiscoveryArmOnly:
            o.taskOptions["targetDiscoveryArmOnly"] == 1,
        targetDiscoveryLowerBodyOnly:
            o.taskOptions["targetDiscoveryLowerBodyOnly"] == 1,
        targetDiscoveryTiedArmKnots:
            o.taskOptions["targetDiscoveryTiedArmKnots"] == 1,
        targetDiscoveryBlockCoordinateSearch:
            o.taskOptions["targetDiscoveryBlockCoordinateSearch"] == 1,
        recedingHorizonSteps: Int(
            o.taskOptions["recedingHorizonSteps"] ?? 0),
        recedingInitialPhaseStep: Int(
            o.taskOptions["recedingInitialPhaseStep"] ?? 0),
        recedingControlHorizonSteps: Int(
            o.taskOptions["recedingControlHorizonSteps"] ?? 1),
        recedingSafetyLookaheadSteps:
            o.taskOptions["recedingSafetyLookaheadSteps"].map(Int.init),
        recedingTerminalHoldSteps: Int(
            o.taskOptions["recedingTerminalHoldSteps"] ?? 0),
        recedingValidationCandidateCount: Int(
            o.taskOptions["recedingValidationCandidateCount"] ?? 4),
        recedingValidationReplicaCount:
            o.taskOptions["recedingValidationReplicaCount"].map(Int.init),
        recedingValidationMinimumSuccessFraction:
            o.taskOptions[
                "recedingValidationMinimumSuccessFraction"] ?? 0.8,
        reconstructionValidationCandidateCount: Int(
            o.taskOptions[
                "reconstructionValidationCandidateCount"] ?? 4),
        reconstructionValidationMinimumSuccessFraction:
            o.taskOptions[
                "reconstructionValidationMinimumSuccessFraction"] ?? 0.8,
        reconstructionOneEnvironmentSearch:
            o.taskOptions["reconstructionOneEnvironmentSearch"] == 1,
        continueTrajectoryFromSourceTerminal:
            o.taskOptions["continueTrajectoryFromSourceTerminal"] == 1,
        preserveProvidedContinuationSeedUpperBody:
            o.taskOptions[
                "preserveProvidedContinuationSeedUpperBody"] == 1,
        recedingShiftInitialSeed:
            o.taskOptions["recedingShiftInitialSeed"] == 1,
        recedingLocomotionBlendProposal:
            o.taskOptions["recedingLocomotionBlendProposal"],
        recedingLocomotionZeroResidualProposal:
            o.taskOptions["recedingLocomotionZeroResidualProposal"] == 1,
        recedingLocomotionCheckpointDirectory:
            o.locomotionCheckpoint,
        recedingLocomotionCommandSpeed:
            o.taskOptions["recedingLocomotionCommandSpeed"],
        recedingForwardOnlyBaseCommand:
            o.taskOptions["recedingForwardOnlyBaseCommand"] == 1,
        recedingHolonomicBaseCommand:
            o.taskOptions["recedingHolonomicBaseCommand"] == 1,
        carryBaseLegActionFractionOverride:
            o.taskOptions["carryBaseLegActionFractionOverride"],
        legBlendKnotCount: legBlendKnotCount,
        legResidualKnotCount: legResidualKnotCount,
        maximumLegResidualAction: maximumLegResidualAction,
        torsoResidualKnotCount: torsoResidualKnotCount,
        maximumTorsoResidualAction: maximumTorsoResidualAction,
        armAsymmetryKnotCount: armAsymmetryKnotCount,
        maximumArmAsymmetryAction: maximumArmAsymmetryAction,
        graspAnchorFeedbackBlend:
            o.taskOptions["graspAnchorFeedbackBlend"] ?? 0,
        graspAnchorFeedbackVelocityHorizonSeconds:
            o.taskOptions[
                "graspAnchorFeedbackVelocityHorizonSeconds"] ?? 0.04,
        graspAnchorFeedbackMaximumActionCorrection:
            o.taskOptions[
                "graspAnchorFeedbackMaximumActionCorrection"] ?? 0.7,
        graspAnchorFeedbackInwardPreloadMeters:
            o.taskOptions[
                "graspAnchorFeedbackInwardPreloadMeters"] ?? 0,
        minimumTargetCarryDistanceMeters:
            o.taskOptions["minimumTargetCarryDistanceMeters"] ?? 0,
        maximumTargetPathCarryRegressionMeters:
            o.taskOptions["maximumTargetPathCarryRegressionMeters"] ?? 0,
        targetDiscoveryObjectiveCarryDistanceMeters:
            o.taskOptions[
                "targetDiscoveryObjectiveCarryDistanceMeters"],
        minimumTargetDestinationProgressMeters:
            o.taskOptions["minimumTargetDestinationProgressMeters"] ?? 0,
        targetDiscoveryObjectiveDestinationProgressMeters:
            o.taskOptions[
                "targetDiscoveryObjectiveDestinationProgressMeters"],
        minimumTargetRootDestinationProgressMeters:
            o.taskOptions[
                "minimumTargetRootDestinationProgressMeters"] ?? 0,
        targetDiscoveryObjectiveRootDestinationProgressMeters:
            o.taskOptions[
                "targetDiscoveryObjectiveRootDestinationProgressMeters"],
        minimumTargetTouchdowns: Int(
            o.taskOptions["minimumTargetTouchdowns"] ?? 0),
        minimumTargetAlternatingSteps: Int(
            o.taskOptions["minimumTargetAlternatingSteps"] ?? 0),
        minimumTargetSwingFootLiftMeters:
            o.taskOptions["minimumTargetSwingFootLiftMeters"] ?? 0,
        targetDiscoveryObjectiveSwingFootLiftMeters:
            o.taskOptions[
                "targetDiscoveryObjectiveSwingFootLiftMeters"],
        minimumTargetFootAirTimeSeconds:
            o.taskOptions["minimumTargetFootAirTimeSeconds"] ?? 0,
        targetDiscoveryObjectiveFootAirTimeSeconds:
            o.taskOptions[
                "targetDiscoveryObjectiveFootAirTimeSeconds"],
        minimumTargetFootUnloadingFraction:
            o.taskOptions["minimumTargetFootUnloadingFraction"] ?? 0,
        targetDiscoveryObjectiveFootUnloadingFraction:
            o.taskOptions[
                "targetDiscoveryObjectiveFootUnloadingFraction"],
        minimumTargetTerminalFootUnloadingFraction:
            o.taskOptions[
                "minimumTargetTerminalFootUnloadingFraction"],
        minimumTargetClearanceMeters:
            o.taskOptions["minimumTargetClearanceMeters"] ?? 0.01,
        targetDiscoveryObjectiveClearanceMeters:
            o.taskOptions[
                "targetDiscoveryObjectiveClearanceMeters"],
        maximumTargetPathDownwardBoxVelocityMPS:
            o.taskOptions[
                "maximumTargetPathDownwardBoxVelocityMPS"],
        minimumTargetTerminalClearanceMeters:
            o.taskOptions["minimumTargetTerminalClearanceMeters"],
        maximumTargetTerminalDownwardBoxVelocityMPS:
            o.taskOptions[
                "maximumTargetTerminalDownwardBoxVelocityMPS"],
        minimumTargetGraspQuality:
            o.taskOptions["minimumTargetGraspQuality"] ?? 0,
        targetDiscoveryObjectiveGraspQuality:
            o.taskOptions["targetDiscoveryObjectiveGraspQuality"],
        targetFeasibilityDwellSteps: Int(
            o.taskOptions["targetFeasibilityDwellSteps"] ?? 1),
        requireStableCarryPath:
            o.taskOptions["requireStableCarryPath"] == 1,
        trajectoryKnotCount: knotCount,
        robustReplayCount: Int(
            o.taskOptions["robustReplayCount"] ?? 32),
        robustActionNoiseStandardDeviation:
            o.taskOptions["robustActionNoiseStandardDeviation"] ?? 0,
        optimizationActionNoiseStandardDeviation:
            o.taskOptions["optimizationActionNoiseStandardDeviation"] ?? 0,
        optimizationActionNoiseReplicaCount: Int(
            o.taskOptions["optimizationActionNoiseReplicaCount"] ?? 1),
        initialStandardDeviation:
            o.taskOptions["initialStandardDeviation"] ?? 0.25,
        eliteFraction: o.taskOptions["eliteFraction"] ?? 0.05,
        seed: o.seed,
        optimizerSeed: o.taskOptions["optimizerSeed"].map {
            UInt64($0)
        })
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let report: HumanoidBoxPhysicalFlowReport
    do {
        report = try HumanoidBoxPhysicalFlowExperiment.run(
            checkpointDirectory: checkpoint,
            targetTrajectory: o.taskOptions["zeroTargetTrajectory"] == 1
                ? [Float](repeating: 0,
                    count: targetArmParameterCount + legBlendKnotCount)
                : targetTrajectory,
            proposalTrajectory: proposalTrajectory,
            archivedTargetTrajectorySequence:
                archivedTargetTrajectorySequence,
            archivedTargetTrajectorySequencePhaseSteps:
                archivedTargetTrajectorySequencePhaseSteps,
            archivedTargetPredictedRecoveryPathSafe:
                archivedTargetPredictedRecoveryPathSafe,
            archivedSourceWarmupActions:
                archivedSourceWarmupActions,
            sourceStages: sourceStages,
            configuration: configuration)
    } catch let failure as HumanoidBoxPhysicalFlowTargetFailure {
        let failureData = try encoder.encode(failure)
        if let output = o.output {
            let url = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try failureData.write(to: url, options: .atomic)
        }
        if o.json { print(String(decoding: failureData, as: UTF8.self)) }
        fail(failure.description)
    }
    let data = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: data, as: UTF8.self))
    } else {
        let finitePrefixPassed = report.targetFinitePrefixGatePassed
            ?? report.targetPlanningGatePassed
        let reusableFrontierPassed = report
            .targetReusableFrontierGatePassed
            ?? (finitePrefixPassed
                && (report.targetPredictedRecoveryPathSafe
                    ?? (report.recedingHorizonSteps == nil)))
        print(String(format:
            "H1 box flow source %dfr %.0f%% target step %d carry %.3fm destination %+0.3fm clearance %.3fm upright %.3f/%.3f",
            report.sourceTrajectorySteps,
            100 * report.sourceReplaySuccessFraction,
            report.selectedTargetStep, report.targetCarryDistanceMeters,
            report.targetDestinationProgressMeters ?? 0,
            report.targetClearanceMeters,
            report.targetRobotUprightAlignment,
            report.targetBoxUprightAlignment))
        for generation in report.generationHistory {
            print(String(format:
                "flow %2d loss %.4f max %.3f median %.4f std %.3f",
                generation.generation, generation.bestLoss,
                generation.bestMaximumNormalizedError,
                generation.medianLoss,
                generation.meanStandardDeviation))
        }
        print(String(format:
            "optimized max %.3f robust %.0f%% replay %.3g finite-prefix %@ reusable-frontier %@ reconstruction %@ overall %@",
            report.optimized.maximumNormalizedError,
            100 * report.robustReplaySuccessFraction,
            report.selectedReplayMaximumNormalizedStateError,
            finitePrefixPassed ? "PASS" : "FAIL",
            reusableFrontierPassed ? "PASS" : "FAIL",
            report.reconstructionGatePassed ? "PASS" : "FAIL",
            report.goGatePassed ? "PASS" : "FAIL"))
    }

case "probe-h1-box-lift":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let checkpoint = o.checkpoint else {
        fail("usage: avbd probe-h1-box-lift --checkpoint DIR "
            + "[--envs N --updates GENERATIONS --frames PROBE_STEPS --json]")
    }
    let initialProbeTrajectory: [Float]? = try {
        guard o.runName != "ppo" else { return nil }
        let prior = try JSONDecoder().decode(
            HumanoidBoxCarryActuationProbeReport.self,
            from: Data(contentsOf: URL(fileURLWithPath: o.runName)))
        return prior.bestArmTarget
    }()
    let configuration = HumanoidBoxCarryActuationProbeConfiguration(
        populationSize: o.envs,
        generations: o.updates,
        maximumWarmupSteps: Int(
            o.taskOptions["maximumWarmupSteps"] ?? 600),
        contactDwellSteps: Int(
            o.taskOptions["contactDwellSteps"] ?? 8),
        probeSteps: o.frames,
        rampSteps: Int(o.taskOptions["rampSteps"] ?? 24),
        trajectoryKnotCount: Int(
            o.taskOptions["trajectoryKnotCount"] ?? 3),
        sustainedCarryObjective:
            o.taskOptions["sustainedCarryObjective"] == 1,
        initialStandardDeviation:
            o.taskOptions["initialStandardDeviation"] ?? 0.30,
        initialArmTrajectory: initialProbeTrajectory,
        eliteFraction: o.taskOptions["eliteFraction"] ?? 0.125,
        seed: o.seed)
    let report = try HumanoidBoxCarryActuationProbe.run(
        checkpointDirectory: checkpoint, configuration: configuration)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
    if o.json {
        print(String(decoding: data, as: UTF8.self))
    } else {
        for generation in report.generations {
            print(String(format:
                "probe %2d warmup %3d score %+.3f rise %+.4fm "
                    + "clearance %+.4fm carry %.3fm goal +%.3fm/%.3fm "
                    + "contact %.1f%% "
                    + "stable %.1f%%/%dfr %@",
                generation.generation, generation.warmupSteps,
                generation.bestScore,
                generation.maximumBoxCenterRiseMeters,
                generation.maximumBoxClearanceMeters,
                generation.maximumCarryDistanceMeters ?? 0,
                generation.maximumDestinationProgressMeters ?? 0,
                generation.minimumPlacementPlanarDistanceMeters ?? .nan,
                generation.bilateralContactFraction * 100,
                generation.unsupportedFraction * 100,
                generation.maximumStableUnsupportedSteps ?? 0,
                generation.physicallyLifted ? "LIFTED" : "planted"))
        }
        print("best bounded arm target ["
            + report.bestArmTarget.map { String(format: "%+.5f", $0) }
                .joined(separator: ",") + "]")
        let passed = configuration.sustainedCarryObjective
            ? report.succeeded == true : report.physicallyLifted
        print(passed
            ? "physical carry feasibility PASS"
            : "physical carry feasibility FAIL")
    }

case "eval-arm-expert":
    let o = parseOptions(Array(args.dropFirst(1)))
    let env = try ArmPushTEnv(
        numEnvironments: o.envs, seed: o.seed,
        linkLength1: o.taskOptions["linkLength1"]
            ?? ArmPushTEnv.linkLengths.x,
        linkLength2: o.taskOptions["linkLength2"]
            ?? ArmPushTEnv.linkLengths.y)
    let ids = Array(0..<o.envs)
    env.reset(ids, seeds: ids.map {
        o.seed &+ UInt64($0) &* 0x9E3779B97F4A7C15
    })
    let expert = ArmPushTGeometricExpert(
        numEnvironments: o.envs,
        contactPreload: o.taskOptions["expertContactPreload"] ?? 0.014)
    let actionSmoothing = o.taskOptions["expertActionSmoothing"] ?? 0.60
    guard actionSmoothing > 0, actionSmoothing <= 1 else {
        fail("expertActionSmoothing must be in (0, 1]")
    }
    var maximumCoverage = [Float](repeating: 0, count: o.envs)
    var firstSuccessStep = [Int?](repeating: nil, count: o.envs)
    var appliedActions = ContiguousArray(repeating: Float(0), count: o.envs * 2)
    for step in 0..<o.frames {
        let states = env.states()
        let requestedActions = expert.actions(environment: env, states: states)
        for i in appliedActions.indices {
            appliedActions[i] += actionSmoothing
                * (requestedActions[i] - appliedActions[i])
        }
        if let watch = o.watch, (0..<o.envs).contains(watch), step % 20 == 0 {
            let s = states[watch]
            print(String(format:
                "step %3d tip (%+.3f,%+.3f) block (%+.3f,%+.3f) yaw %+.3f action (%+.3f,%+.3f) coverage %.3f",
                step, s.tipPosition.x, s.tipPosition.y,
                s.blockPosition.x, s.blockPosition.y, s.blockYaw,
                appliedActions[watch * 2], appliedActions[watch * 2 + 1],
                env.coverage(watch, state: s)))
        }
        // Match ArmPushTTask's maintained 20 Hz controller over the 120 Hz
        // physics step so this diagnostic has the same five-second horizon.
        env.step(normalizedActions: appliedActions, decimation: 6)
        let next = env.states()
        for e in 0..<o.envs {
            let coverage = env.coverage(e, state: next[e])
            maximumCoverage[e] = max(maximumCoverage[e], coverage)
            if coverage > ArmPushTEnv.successCoverage,
               firstSuccessStep[e] == nil {
                firstSuccessStep[e] = step + 1
            }
        }
    }
    let successes = firstSuccessStep.compactMap { $0 }
    let sortedCoverage = maximumCoverage.sorted()
    let medianCoverage = sortedCoverage[sortedCoverage.count / 2]
    let meanCoverage = maximumCoverage.reduce(0, +) / Float(o.envs)
    let meanSuccessStep = successes.isEmpty ? 0
        : Float(successes.reduce(0, +)) / Float(successes.count)
    print(String(format:
        "arm expert  envs %d  success %d (%.1f%%)  coverage mean %.3f median %.3f  success_step %.1f",
        o.envs, successes.count, 100 * Float(successes.count) / Float(o.envs),
        meanCoverage, medianCoverage, meanSuccessStep))

case "experiment-pusht-flow":
    let o = parseOptions(Array(args.dropFirst(1)))
    let report = try PushTPhysicalFlowExperiment.run(
        configuration: makePushTPhysicalFlowConfiguration(
            o, arguments: args))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(report)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try reportData.write(to: url, options: .atomic)
    }
    print(String(decoding: reportData, as: UTF8.self))

case "eval-pusht-flow":
    let o = parseOptions(Array(args.dropFirst(1)))
    let seedStride = UInt64(max(
        1, Int(o.taskOptions["seedStride"] ?? 104_729)))
    var reports = [PushTPhysicalFlowReport]()
    for index in 0..<max(1, o.episodes) {
        var configuration = makePushTPhysicalFlowConfiguration(
            o, arguments: args, defaultGenerations: 2)
        configuration.seed = o.seed &+ UInt64(index) &* seedStride
        let report = try PushTPhysicalFlowExperiment.run(
            configuration: configuration)
        reports.append(report)
        print(String(format:
            "geometric flow %3d/%3d seed %llu  objective %.5g  legacy %.5g  robust %.0f%%  go %d",
            index + 1, max(1, o.episodes), configuration.seed,
            report.optimizedSpline.loss,
            report.optimizedSpline.legacyWeightedLoss,
            100 * report.robustReplaySuccessFraction,
            report.goGatePassed ? 1 : 0))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(reports)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try reportData.write(to: url, options: .atomic)
    } else {
        print(String(decoding: reportData, as: UTF8.self))
    }

case "collect-pusht-flow-teacher":
    let o = parseOptions(Array(args.dropFirst(1)))
    guard let output = o.output else {
        fail("usage: avbd collect-pusht-flow-teacher --output <dataset.json> "
            + "[--episodes N] [physical-flow options]")
    }
    let configuration = makePushTPhysicalFlowConfiguration(
        o, arguments: args, defaultGenerations: 10)
    let dataset = try PushTPhysicalFlowDatasetBuilder.collect(
        requestedSamples: max(1, o.episodes),
        seedStart: o.seed,
        seedStride: UInt64(max(
            1, Int(o.taskOptions["seedStride"] ?? 104_729))),
        experimentConfiguration: configuration)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(dataset)
    let url = URL(fileURLWithPath: output)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true)
    try data.write(to: url, options: .atomic)
    print("flow dataset -> \(output)  accepted \(dataset.samples.count)/"
        + "\(dataset.requestedSamples)  rejected \(dataset.rejected.count)")

case "train-pusht-flow-proposal":
    guard args.count > 1, !args[1].hasPrefix("--") else {
        fail("usage: avbd train-pusht-flow-proposal <dataset.json> "
            + "--output <checkpoint-dir> [--epochs N] [--hidden N] [--lr R]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    guard let output = o.output else {
        fail("train-pusht-flow-proposal requires --output <checkpoint-dir>")
    }
    let dataset = try JSONDecoder().decode(
        PushTPhysicalFlowDataset.self,
        from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    let report = try PushTPhysicalFlowProposalTrainer.train(
        dataset: dataset,
        epochs: args.contains("--epochs") ? max(1, o.epochs) : 2_000,
        hiddenDimension: args.contains("--hidden") ? max(16, o.hidden) : 128,
        learningRate: o.lr,
        seed: o.seed,
        checkpointDirectory: output)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(report)
    try reportData.write(
        to: URL(fileURLWithPath: "\(output)/training-report.json"),
        options: .atomic)
    print(String(decoding: reportData, as: UTF8.self))

case "eval-pusht-flow-proposal":
    guard args.count > 1, !args[1].hasPrefix("--") else {
        fail("usage: avbd eval-pusht-flow-proposal <checkpoint-dir> "
            + "[--episodes N] [--output reports.json] [physical-flow options]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let proposal = try PushTPhysicalFlowMLXProposal(
        checkpointDirectory: args[1])
    let seedStride = UInt64(max(
        1, Int(o.taskOptions["seedStride"] ?? 104_729)))
    var reports = [PushTPhysicalFlowReport]()
    for index in 0..<max(1, o.episodes) {
        var configuration = makePushTPhysicalFlowConfiguration(
            o, arguments: args, defaultGenerations: 2)
        configuration.seed = o.seed &+ UInt64(index) &* seedStride
        let report = try PushTPhysicalFlowExperiment.run(
            configuration: configuration,
            proposalProvider: { try proposal($0) })
        reports.append(report)
        print(String(format:
            "learned flow %3d/%3d seed %llu  initial %.5g  refined %.5g  robust %.0f%%  go %d",
            index + 1, max(1, o.episodes), configuration.seed,
            report.initialProposalSpline.loss, report.optimizedSpline.loss,
            100 * report.robustReplaySuccessFraction,
            report.goGatePassed ? 1 : 0)
            + "  selected \(report.generationZeroSelectedProposal.rawValue)")
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(reports)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try reportData.write(to: url, options: .atomic)
    } else {
        print(String(decoding: reportData, as: UTF8.self))
    }

case "eval-pusht-flow-retrieval":
    guard args.count > 1, !args[1].hasPrefix("--") else {
        fail("usage: avbd eval-pusht-flow-retrieval <dataset.json> "
            + "[--episodes N] [--output reports.json] [physical-flow options]")
    }
    let o = parseOptions(Array(args.dropFirst(2)))
    let dataset = try JSONDecoder().decode(
        PushTPhysicalFlowDataset.self,
        from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    let proposal = try PushTPhysicalFlowNearestNeighborProposal(
        dataset: dataset)
    let seedStride = UInt64(max(
        1, Int(o.taskOptions["seedStride"] ?? 104_729)))
    var reports = [PushTPhysicalFlowReport]()
    for index in 0..<max(1, o.episodes) {
        var configuration = makePushTPhysicalFlowConfiguration(
            o, arguments: args, defaultGenerations: 2)
        configuration.seed = o.seed &+ UInt64(index) &* seedStride
        let report = try PushTPhysicalFlowExperiment.run(
            configuration: configuration,
            proposalProvider: { proposal($0) })
        reports.append(report)
        print(String(format:
            "retrieved flow %3d/%3d seed %llu  neighbor %llu d %.3f  initial %.5g  refined %.5g  robust %.0f%%  go %d",
            index + 1, max(1, o.episodes), configuration.seed,
            proposal.lastNeighborSeed ?? 0,
            proposal.lastNormalizedDistance ?? .infinity,
            report.initialProposalSpline.loss, report.optimizedSpline.loss,
            100 * report.robustReplaySuccessFraction,
            report.goGatePassed ? 1 : 0)
            + "  selected \(report.generationZeroSelectedProposal.rawValue)")
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(reports)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try reportData.write(to: url, options: .atomic)
    } else {
        print(String(decoding: reportData, as: UTF8.self))
    }

case "select-rl":
    let raw = Array(args.dropFirst())
    let o = parseOptions(raw)
    var reportPaths = [String]()
    var index = 0
    while index < raw.count {
        if raw[index] == "--output" {
            index += 2
        } else {
            reportPaths.append(raw[index])
            index += 1
        }
    }
    guard !reportPaths.isEmpty else {
        fail("usage: avbd select-rl <validation-report.json...> "
            + "[--output selection.json]")
    }
    let reports = try reportPaths.map {
        try JSONDecoder().decode(
            PPOEvaluationMetrics.self,
            from: Data(contentsOf: URL(fileURLWithPath: $0)))
    }
    let selection = try PPOCheckpointSelection.make(reports)
    let selectionEncoder = JSONEncoder()
    selectionEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let selectionData = try selectionEncoder.encode(selection)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try selectionData.write(to: url, options: .atomic)
    }
    print(String(decoding: selectionData, as: UTF8.self))

case "verify-selection-rl":
    let raw = Array(args.dropFirst())
    guard raw.count >= 2 else {
        fail("usage: avbd verify-selection-rl <selection.json> <test-report.json...>")
    }
    let selection = try JSONDecoder().decode(
        PPOCheckpointSelection.self,
        from: Data(contentsOf: URL(fileURLWithPath: raw[0])))
    let reports = try raw.dropFirst().map {
        try JSONDecoder().decode(
            PPOEvaluationMetrics.self,
            from: Data(contentsOf: URL(fileURLWithPath: $0)))
    }
    for report in reports { try selection.validateTestReport(report) }
    let validationSeeds = selection.validationSeeds ?? [selection.validationSeed]
    print("verified \(reports.count) test report(s) for immutable checkpoint "
        + "\(selection.selectedCheckpointFingerprint); validation seeds "
        + "\(validationSeeds.map(String.init).joined(separator: ",")) "
        + "are absent from test reports")

case "aggregate-rl":
    let raw = Array(args.dropFirst())
    let o = parseOptions(raw)
    var reportPaths = [String]()
    var index = 0
    while index < raw.count {
        if raw[index] == "--output" {
            index += 2
        } else {
            reportPaths.append(raw[index])
            index += 1
        }
    }
    guard !reportPaths.isEmpty else {
        fail("usage: avbd aggregate-rl <report.json...> [--output summary.json]")
    }
    let reports = try reportPaths.map {
        try JSONDecoder().decode(
            PPOEvaluationMetrics.self,
            from: Data(contentsOf: URL(fileURLWithPath: $0)))
    }
    let aggregate = try PPOEvaluationAggregate.make(reports)
    let aggregateEncoder = JSONEncoder()
    aggregateEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let aggregateData = try aggregateEncoder.encode(aggregate)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try aggregateData.write(to: url, options: .atomic)
    }
    print(String(decoding: aggregateData, as: UTF8.self))
    if !aggregate.publishable { exit(2) }

case "aggregate-checkpoint-rl":
    let raw = Array(args.dropFirst())
    let o = parseOptions(raw)
    var reportPaths = [String]()
    var index = 0
    while index < raw.count {
        if raw[index] == "--output" {
            index += 2
        } else {
            reportPaths.append(raw[index])
            index += 1
        }
    }
    guard !reportPaths.isEmpty else {
        fail("usage: avbd aggregate-checkpoint-rl <report.json...> "
            + "[--output summary.json]")
    }
    let reports = try reportPaths.map {
        try JSONDecoder().decode(
            PPOEvaluationMetrics.self,
            from: Data(contentsOf: URL(fileURLWithPath: $0)))
    }
    let aggregate = try PPOCheckpointEvaluationAggregate.make(reports)
    let aggregateEncoder = JSONEncoder()
    aggregateEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let aggregateData = try aggregateEncoder.encode(aggregate)
    if let output = o.output {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try aggregateData.write(to: url, options: .atomic)
    }
    print(String(decoding: aggregateData, as: UTF8.self))
    if !aggregate.robustAcrossEvaluationSeeds { exit(2) }

case "run":
    guard args.count > 1 else { fail("usage: avbd run <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)

    if o.useCPU {
        let solver = scene.makeCPUSolver()
        let t0 = Date()
        for f in 0..<o.frames {
            solver.step()
            if !o.json && (f + 1) % o.statsEvery == 0 {
                let err = solver.maxConstraintError()
                print(String(format: "frame %5d  err %.5f  forces %d", f + 1, err, solver.forces.count))
            }
        }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
        let err = solver.maxConstraintError()
        if o.json {
            print("{\"backend\":\"cpu\",\"demo\":\"\(scene.name)\",\"frames\":\(o.frames),\"msPerFrame\":\(ms),\"maxConstraintError\":\(err)}")
        } else {
            print(String(format: "cpu: %d bodies, %.3f ms/frame, final err %.5f",
                         scene.bodies.count, ms, err))
        }
    } else {
        let solver = try GPUSolver(scene: scene)
        let t0 = Date()
        for f in 0..<o.frames {
            solver.step()
            if !o.json && (f + 1) % o.statsEvery == 0 {
                let err = solver.maxConstraintError()
                var extra = ""
                if let w = o.watch {
                    let p = solver.bodyPosition(w)
                    extra = String(format: "  body%d (%.3f, %.3f, %.3f)", w, p.x, p.y, p.z)
                }
                let colors = solver.lastColorCounts.filter { $0 > 0 }.count
                print(String(format: "frame %5d  err %.5f  pairs %5d  colors %2d%@",
                             f + 1, err, solver.lastNumPairs, colors, extra))
            }
        }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
        let err = solver.maxConstraintError()
        if o.json {
            print("{\"backend\":\"gpu\",\"demo\":\"\(scene.name)\",\"bodies\":\(scene.bodies.count),\"frames\":\(o.frames),\"msPerFrame\":\(ms),\"maxConstraintError\":\(err),\"pairs\":\(solver.lastNumPairs)}")
        } else {
            print(String(format: "gpu: %d bodies, %.3f ms/frame, final err %.5f, %d pairs",
                         scene.bodies.count, ms, err, solver.lastNumPairs))
        }
    }

case "rodexp":
    // Inextensibility experiment: flagwhip under three structural-edge
    // regimes. Reports per-window KE maxima (envelope must decay), worst
    // structural stretch, and the rod dual magnitude.
    let o = parseOptions(Array(args.dropFirst(1)))
    let variants: [(String, Float, Bool, Float)] = [
        ("stiff-5k", 5000, false, 0),          // current default (toy: stretches)
        ("stiff-2e5", 2e5, false, 0),          // honest stiffness candidate
        ("hard-rods", 0, true, 0),             // AL rods, no decay (pump repro)
        ("hard-decay", 0, true, 16),           // AL rods + rotation decay
    ]
    let frames = o.frames > 300 ? o.frames : 3600
    for (name, k, hard, decay) in variants {
        var scene = Demos.flagwhip(res: o.res ?? 16,
                                   structuralK: hard ? 5000 : k,
                                   hardRods: hard)
        scene.settings.rodDecayPow = decay
        if let it = o.iterations { scene.settings.iterations = it }
        let solver = try GPUSolver(scene: scene)
        var windowMax: [Float] = []
        var cur: Float = 0
        var worstStretch: Float = 0
        for f in 0..<frames {
            solver.step()
            if f % 10 == 0 {
                var ke: Float = 0
                for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                    ke += 0.5 * solver.bodyMass(b) * length_squared(solver.bodyVelocity(b))
                }
                cur = max(cur, ke)
            }
            if (f + 1) % 600 == 0 {
                windowMax.append(cur)
                cur = 0
                if f > frames / 2 {
                    let (_, st) = solver.debugClothMetrics()
                    worstStretch = max(worstStretch, st)
                }
            }
        }
        let envelope = windowMax.map { String(format: "%.3f", $0) }.joined(separator: " ")
        let growing = windowMax.count >= 3
            && windowMax.last! > 1.5 * windowMax[1]
        print(String(format: "%@: KE windows [%@] %@  stretch %.4f",
                     name, envelope, growing ? "GROWING(PUMP)" : "decaying",
                     worstStretch))
    }

case "profile":
    // Per-stage GPU time breakdown (encoder-boundary timestamps).
    guard args.count > 1 else { fail("usage: avbd profile <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let solver = try GPUSolver(scene: scene)
    print("bodies \(scene.bodies.count)  tris \(scene.tris.count)+\(solver.tetBoundaryTris.count)b  springs \(scene.springs.count)  joints \(scene.joints.count)  colors \(solver.staticUsedColors)  persistent-capacity \(solver.persistentCapacity)")
    for _ in 0..<30 { solver.step() }      // warm up
    solver.sync()
    solver.profiling = true
    solver.resetProfile()
    let t0 = Date()
    for _ in 0..<o.frames { solver.step() }
    solver.sync()
    let wallMS = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
    let total = solver.profileNS.values.reduce(0, +)
    print(String(format: "wall %.3f ms/frame  (gpu-stage sum %.3f ms)",
                 wallMS, total / Double(o.frames) / 1e6))
    for (name, ns) in solver.profileNS.sorted(by: { $0.value > $1.value }) {
        let ms = ns / Double(o.frames) / 1e6
        print(String(format: "  %-22s %8.3f ms  %5.1f%%", (name as NSString).utf8String!,
                     ms, ns / total * 100))
    }

case "clothgate":
    // Cloth gate runner: step a demo and report element-contact metrics
    // (worst V-T clearance, worst structural stretch, soft contact count).
    guard args.count > 1 else { fail("usage: avbd clothgate <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let solver = try GPUSolver(scene: scene)
    print("bodies \(scene.bodies.count)  tris \(scene.tris.count)  persistent-capacity \(solver.persistentCapacity)")
    var worstGap: Float = .greatestFiniteMagnitude
    var worstStretch: Float = 0
    var ke: Float = 0
    for f in 0..<o.frames {
        solver.step()
        if (f + 1) % o.statsEvery == 0 || f == o.frames - 1 {
            let (gap, stretch) = solver.debugClothMetrics()
            // KE over particles (cloth energy envelope) + fastest node
            ke = 0
            var vmax: Float = 0
            var fastest = -1
            for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                let v = solver.bodyVelocity(b)
                let s2 = length_squared(v)
                ke += 0.5 * solver.bodyMass(b) * s2
                if s2 > vmax * vmax { vmax = s2.squareRoot(); fastest = b }
            }
            var loc = ""
            if fastest >= 0 {
                let p = solver.bodyPosition(fastest)
                loc = String(format: "  vmax %6.2f @%d (%.2f, %.2f, %.2f)",
                             vmax, fastest, p.x, p.y, p.z)
            }
            if ProcessInfo.processInfo.environment["AVBD_ZONES"] != nil {
                var zk = [Float](repeating: 0, count: 3)
                for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                    let z = solver.bodyPosition(b).z
                    let v = solver.bodyVelocity(b)
                    let e = 0.5 * solver.bodyMass(b) * length_squared(v)
                    zk[z < 0.3 ? 0 : (z < 1.0 ? 1 : 2)] += e
                }
                loc += String(format: "  zones KE [pool %.3f skirt %.3f crown %.3f]",
                              zk[0], zk[1], zk[2])
            }
            if f > o.frames / 2 {       // settled-half metrics
                worstGap = min(worstGap, gap)
                worstStretch = max(worstStretch, stretch)
            }
            print(String(format: "frame %5d  gap %+.4f  stretch %.4f  soft %5d  pairs %5d  KE %.4f%@",
                         f + 1, gap, stretch, solver.lastNumSoft, solver.lastNumPairs, ke, loc))
        }
    }
    print(String(format: "settled-half: worstGap %+.4f  worstStretch %.4f",
                 worstGap == .greatestFiniteMagnitude ? 0 : worstGap, worstStretch))
    let (wa, wb) = solver.lastWorstSpring
    if wa >= 0 {
        let p1 = solver.bodyPosition(wa), p2 = solver.bodyPosition(wb)
        print(String(format: "worst spring %d-%d  (%.2f,%.2f,%.2f) - (%.2f,%.2f,%.2f)",
                     wa, wb, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z))
        if solver.lastWorstSpringIdx >= 0 {
            let (lam, pen, c0, rest) = solver.debugSpringDual(solver.lastWorstSpringIdx)
            print(String(format: "  dual: lambda %.3f  penalty %.1f  C0 %.4f  rest %.4f  len %.4f",
                         lam, pen, c0, rest, distance(p1, p2)))
        }
    }

case "collect":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.collect(envs: o.envs, steps: o.frames,
                              path: "runs/pusht/data", bc: args.contains("--bc"))

case "train-bc":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.trainBC(dataPath: "runs/pusht/data", iters: o.frames,
                              batch: o.batch, latent: o.latent, lr: o.lr,
                              modelPath: "runs/pusht/model")

case "collect-dagger":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.collectDagger(envs: o.envs, steps: o.frames,
                                    path: "runs/pusht/data",
                                    policyPath: "runs/pusht/model",
                                    latent: o.latent)

case "train-ppo":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PPOPipeline.train(envs: o.envs, updates: o.updates, horizon: o.horizon,
                          epochs: o.epochs, minibatch: o.batch,
                          lr: args.contains("--lr") ? o.lr : 3e-5,
                          latent: args.contains("--latent") ? o.latent : 192,
                          initFrom: args.contains("--scratch")
                              ? nil : "runs/pusht/model/bc.safetensors")

case "solve-ppo":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PPOPipeline.solve(modelPath: "runs/pusht/model",
                          episodes: o.episodes,
                          latent: args.contains("--latent") ? o.latent : 192)

case "train-lawm":
    let o = parseOptions(Array(args.dropFirst(1)))
    try LatentActionPipeline.train(dataPath: "runs/pusht/data", iters: o.frames,
                                   batch: o.batch, latent: o.latent, lr: o.lr,
                                   modelPath: "runs/pusht/model")

case "solve-lawm":
    let o = parseOptions(Array(args.dropFirst(1)))
    try LatentActionPipeline.solve(modelPath: "runs/pusht/model",
                                   episodes: o.episodes, latent: o.latent)

case "solve-bc":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.solveBC(modelPath: "runs/pusht/model",
                              episodes: o.episodes, latent: o.latent)

case "train-wm":
    let o = parseOptions(Array(args.dropFirst(1)))
    if args.contains("--ensemble") {
        for m in 0..<3 {
            print("=== ensemble member \(m) ===")
            try PushTPipeline.train(dataPath: "runs/pusht/data", iters: o.frames,
                                    batch: o.batch, latent: o.latent, lr: o.lr,
                                    lambda: o.lambda, modelPath: "runs/pusht/model",
                                    member: m)
        }
    } else {
        try PushTPipeline.train(dataPath: "runs/pusht/data", iters: o.frames,
                                batch: o.batch, latent: o.latent, lr: o.lr,
                                lambda: o.lambda, modelPath: "runs/pusht/model")
    }

case "probe-wm":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.probe(dataPath: "runs/pusht/data",
                            modelPath: "runs/pusht/model", latent: o.latent)

case "oracle-pusht":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.oracle(episodes: o.episodes)

case "solve-pusht":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.solve(modelPath: "runs/pusht/model", episodes: o.episodes,
                            seed: UInt64(o.watch ?? 11),
                            latent: o.latent, debug: args.contains("--debug"),
                            oracleNull: args.contains("--oracle-null"))

case "bench":
    guard args.count > 1 else { fail("usage: avbd bench <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let solver = try GPUSolver(scene: scene)
    // warmup
    for _ in 0..<10 { solver.step() }
    if args.contains("--capture") {
        let mgr = MTLCaptureManager.shared()
        let cd = MTLCaptureDescriptor()
        cd.captureObject = solver.device
        cd.destination = .gpuTraceDocument
        cd.outputURL = URL(fileURLWithPath: "avbd-\(scene.name).gputrace")
        try mgr.startCapture(with: cd)
        for _ in 0..<3 { solver.step() }
        mgr.stopCapture()
        print("wrote avbd-\(scene.name).gputrace (open in Xcode)")
    }
    solver.profiling = args.contains("--profile")
    solver.resetProfile()
    let t0 = Date()
    var encodeS = 0.0
    let syncEach = args.contains("--syncstep")
    for _ in 0..<o.frames {
        let e0 = Date()
        solver.step()
        if syncEach { solver.sync() }
        encodeS += Date().timeIntervalSince(e0)
    }
    solver.sync()
    let ms = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
    print(String(format: "  cpu encode: %.3f ms/frame", encodeS * 1000 / Double(o.frames)))
    print(String(format: "%@: %d bodies, %d iterations, %.3f ms/frame (%.1f FPS)",
                 scene.name, scene.bodies.count, scene.settings.iterations, ms, 1000 / ms))
    if solver.profiling, solver.profileFrames > 0 {
        let n = Double(solver.profileFrames)
        let total = solver.profileNS.values.reduce(0, +)
        print(String(format: "GPU stage breakdown (%d frames, %.3f ms GPU/frame):",
                     solver.profileFrames, total / n / 1e6))
        for (name, ns) in solver.profileNS.sorted(by: { $0.value > $1.value }) {
            print(String(format: "  %-20s %8.3f ms  %5.1f%%",
                         (name as NSString).utf8String!, ns / n / 1e6,
                         ns / total * 100))
        }
        print(String(format: "  pairs %d  colors %d",
                     solver.lastNumPairs, solver.lastMaxColorUsed + 1))
    }

case "parity":
    guard args.count > 1 else { fail("usage: avbd parity <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let cpu = scene.makeCPUSolver()
    let gpu = try GPUSolver(scene: scene)
    var maxDiff: Float = 0
    for f in 0..<o.frames {
        cpu.step()
        gpu.step()
        var diff: Float = 0
        for i in 0..<scene.bodies.count {
            diff = max(diff, length(cpu.bodies[i].positionLin - gpu.bodyPosition(i)))
        }
        maxDiff = max(maxDiff, diff)
        if (f + 1) % o.statsEvery == 0 {
            print(String(format: "frame %5d  maxDiff %.5f", f + 1, diff))
        }
    }
    print(String(format: "max position divergence over %d frames: %.5f", o.frames, maxDiff))

default:
    fail("unknown command '\(command)'")
}
