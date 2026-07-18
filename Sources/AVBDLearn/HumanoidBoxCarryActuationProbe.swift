import AVBDCore
import Foundation
import simd

/// Configuration for the batched, simulator-in-the-loop grasp-to-lift probe.
///
/// The probe is deliberately not a task controller. It replays a learned MLX
/// policy until the policy has established bilateral physical contact, then
/// uses independent simulator replicas to test bounded, constant arm targets.
/// Its purpose is to separate an exploration/reward failure from an actuator,
/// geometry, or contact-model failure before another expensive PPO run.
public struct HumanoidBoxCarryActuationProbeConfiguration: Sendable {
    public var populationSize: Int
    public var generations: Int
    public var maximumWarmupSteps: Int
    public var contactDwellSteps: Int
    public var probeSteps: Int
    public var rampSteps: Int
    public var trajectoryKnotCount: Int
    public var sustainedCarryObjective: Bool
    public var initialStandardDeviation: Float
    public var initialArmTrajectory: [Float]?
    public var eliteFraction: Float
    public var seed: UInt64

    public init(
        populationSize: Int = 128,
        generations: Int = 6,
        maximumWarmupSteps: Int = 600,
        contactDwellSteps: Int = 8,
        probeSteps: Int = 100,
        rampSteps: Int = 24,
        trajectoryKnotCount: Int = 3,
        sustainedCarryObjective: Bool = false,
        initialStandardDeviation: Float = 0.30,
        initialArmTrajectory: [Float]? = nil,
        eliteFraction: Float = 0.125,
        seed: UInt64 = 1
    ) {
        self.populationSize = populationSize
        self.generations = generations
        self.maximumWarmupSteps = maximumWarmupSteps
        self.contactDwellSteps = contactDwellSteps
        self.probeSteps = probeSteps
        self.rampSteps = rampSteps
        self.trajectoryKnotCount = trajectoryKnotCount
        self.sustainedCarryObjective = sustainedCarryObjective
        self.initialStandardDeviation = initialStandardDeviation
        self.initialArmTrajectory = initialArmTrajectory
        self.eliteFraction = eliteFraction
        self.seed = seed
    }

    func validate() throws {
        guard populationSize >= 4,
              generations > 0,
              maximumWarmupSteps > 0,
              contactDwellSteps > 0,
              probeSteps > 0,
              rampSteps > 0,
              trajectoryKnotCount > 0,
              initialStandardDeviation > 0,
              (initialArmTrajectory == nil
                || initialArmTrajectory?.count == 4 * trajectoryKnotCount),
              initialArmTrajectory?.allSatisfy({ $0.isFinite }) != false,
              eliteFraction > 0,
              eliteFraction <= 0.5 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid box-carry actuation probe configuration")
        }
    }
}

public struct HumanoidBoxCarryActuationProbeGeneration: Codable, Sendable {
    public var generation: Int
    public var warmupSteps: Int
    public var bestScore: Float
    public var bestArmTarget: [Float]
    public var maximumBoxCenterRiseMeters: Float
    public var maximumBoxClearanceMeters: Float
    public var bilateralContactFraction: Float
    public var unsupportedFraction: Float
    public var minimumBoxUprightAlignment: Float
    public var physicallyLifted: Bool
    public var failed: Bool
    public var maximumCarryDistanceMeters: Float?
    public var maximumStableUnsupportedSteps: Int?
    public var minimumPlacementPlanarDistanceMeters: Float?
    public var maximumDestinationProgressMeters: Float?
    public var destinationContactFraction: Float?
    public var succeeded: Bool?
}

public struct HumanoidBoxCarryActuationProbeReport: Codable, Sendable {
    public var checkpointDirectory: String
    public var checkpointTaskRevision: Int
    public var populationSize: Int
    public var generations: [HumanoidBoxCarryActuationProbeGeneration]
    public var bestGeneration: Int
    public var bestScore: Float
    public var bestArmTarget: [Float]
    public var maximumBoxCenterRiseMeters: Float
    public var maximumBoxClearanceMeters: Float
    public var physicallyLifted: Bool
    public var maximumCarryDistanceMeters: Float?
    public var maximumStableUnsupportedSteps: Int?
    public var minimumPlacementPlanarDistanceMeters: Float?
    public var maximumDestinationProgressMeters: Float?
    public var destinationContactFraction: Float?
    public var succeeded: Bool?
}

/// Simulator-in-the-loop feasibility audit for the H1 pickup transition.
public enum HumanoidBoxCarryActuationProbe {
    private static let firstArmAction = 11
    private static let armActionCount = 8

    private struct CandidateResult {
        var score: Float
        var armTarget: [Float]
        var maximumCenterRise: Float
        var maximumClearance: Float
        var bilateralFraction: Float
        var unsupportedFraction: Float
        var minimumUpright: Float
        var physicallyLifted: Bool
        var failed: Bool
        var maximumCarryDistance: Float
        var maximumStableUnsupportedSteps: Int
        var minimumPlacementPlanarDistance: Float
        var maximumDestinationProgress: Float
        var destinationContactFraction: Float
        var succeeded: Bool
    }

    /// Project independent left/right arm targets onto H1's sagittal bilateral
    /// symmetry. A symmetric pinch applies matched vertical motion and
    /// opposing lateral force; without this projection CEM is rewarded for
    /// lifting one corner while the other remains planted on the pedestal.
    static func bilaterallySymmetricArmTarget(_ target: [Float]) -> [Float] {
        precondition(target.count == armActionCount)
        let shoulderPitch = 0.5 * (target[0] + target[4])
        let shoulderRoll = 0.5 * (target[1] - target[5])
        let shoulderYaw = 0.5 * (target[2] - target[6])
        let elbow = 0.5 * (target[3] + target[7])
        return [
            shoulderPitch, shoulderRoll, shoulderYaw, elbow,
            shoulderPitch, -shoulderRoll, -shoulderYaw, elbow,
        ].map { simd_clamp($0, -0.999, 0.999) }
    }

    public static func run(
        checkpointDirectory: String,
        configuration: HumanoidBoxCarryActuationProbeConfiguration = .init()
    ) throws -> HumanoidBoxCarryActuationProbeReport {
        try configuration.validate()
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        guard runner.metadata.task == "humanoid-box-carry-v0" else {
            throw RLEnvironmentError.invalidConfiguration(
                "actuation probe requires a humanoid-box-carry-v0 checkpoint")
        }
        guard let semanticOptions = runner.metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "actuation probe requires checkpoint task configuration metadata")
        }
        let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: runner.metadata.task,
            semanticOptions: semanticOptions,
            maxEpisodeSteps: runner.metadata.maxEpisodeSteps,
            controlDecimation: runner.metadata.controlDecimation)

        var generator = ProbeRandomNumberGenerator(seed: configuration.seed)
        let searchDimension = 4 * configuration.trajectoryKnotCount
        var searchMean = configuration.initialArmTrajectory
            ?? [Float](repeating: 0, count: searchDimension)
        var searchStandardDeviation = [Float](
            repeating: configuration.initialStandardDeviation,
            count: searchDimension)
        var generationReports = [HumanoidBoxCarryActuationProbeGeneration]()
        var overallBest: CandidateResult?
        var overallBestGeneration = 0

        for generation in 0..<configuration.generations {
            let anyTask = try BuiltInRLTasks.registry.make(
                runner.metadata.task,
                configuration: RLTaskConfiguration(
                    numEnvironments: configuration.populationSize,
                    seed: configuration.seed,
                    autoReset: false,
                    options: replayOptions))
            guard let task = anyTask as? HumanoidBoxCarryTask else {
                throw RLEnvironmentError.invalidConfiguration(
                    "registered box-carry task has an unexpected implementation")
            }
            let compatibility = runner.metadata.compatibilityMismatches(
                with: task.spec)
            guard compatibility.isEmpty else {
                throw RLEnvironmentError.invalidConfiguration(
                    "actuation probe checkpoint/task mismatch: "
                        + compatibility.joined(separator: "; "))
            }

            var observation = try task.reset(
                seed: configuration.seed &+ UInt64(generation))
            var result = RLStepBatch(spec: task.spec)
            var contactStreak = 0
            var warmupSteps = 0
            while warmupSteps < configuration.maximumWarmupSteps,
                  contactStreak < configuration.contactDwellSteps {
                let actions = try policyActions(
                    runner: runner, task: task, observation: observation)
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
            guard contactStreak >= configuration.contactDwellSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "checkpoint did not establish a stable bilateral grasp "
                        + "within \(configuration.maximumWarmupSteps) steps")
            }

            let candidates = sampledTargets(
                count: configuration.populationSize,
                mean: searchMean,
                standardDeviation: searchStandardDeviation,
                generator: &generator)
            let initialManipulation = task.environment.manipulationStates()
            let destinationTarget = task.currentPlacementTarget(environment: 0)
            let initialPlacementPlanarDistance = hypot(
                initialManipulation[0].object.position.x - destinationTarget.x,
                initialManipulation[0].object.position.y - destinationTarget.y)
            let initialClearances = result.metrics["state/box_clearance_m"]!
            var maximumCenterRise = [Float](
                repeating: -.infinity, count: configuration.populationSize)
            var maximumClearance = [Float](initialClearances)
            var bilateralSteps = [Int](
                repeating: 0, count: configuration.populationSize)
            var unsupportedSteps = [Int](
                repeating: 0, count: configuration.populationSize)
            var consecutiveStableUnsupportedSteps = [Int](
                repeating: 0, count: configuration.populationSize)
            var maximumStableUnsupportedSteps = [Int](
                repeating: 0, count: configuration.populationSize)
            var positiveClearanceSums = [Float](
                repeating: 0, count: configuration.populationSize)
            var stablePositiveClearanceSums = [Float](
                repeating: 0, count: configuration.populationSize)
            var minimumUpright = [Float](
                repeating: 1, count: configuration.populationSize)
            var physicallyLifted = [Bool](
                repeating: false, count: configuration.populationSize)
            var failed = [Bool](
                repeating: false, count: configuration.populationSize)
            var succeeded = [Bool](
                repeating: false, count: configuration.populationSize)
            var maximumCarryDistance = [Float](
                repeating: 0, count: configuration.populationSize)
            var minimumPlacementPlanarDistance = [Float](
                repeating: initialPlacementPlanarDistance,
                count: configuration.populationSize)
            var destinationContactSteps = [Int](
                repeating: 0, count: configuration.populationSize)

            for probeStep in 0..<configuration.probeSteps {
                var actions = try policyActions(
                    runner: runner, task: task, observation: observation)
                let progress = Float(probeStep + 1)
                    / Float(configuration.probeSteps)
                for environment in 0..<configuration.populationSize {
                    let actionBase = environment * task.spec.action.elementCount
                    let delta = trajectoryArmDelta(
                        candidates[environment],
                        knotCount: configuration.trajectoryKnotCount,
                        progress: progress)
                    for arm in 0..<armActionCount {
                        let index = actionBase + firstArmAction + arm
                        actions.values[index] = simd_clamp(
                            actions.values[index] + delta[arm],
                            -0.999, 0.999)
                    }
                }
                try task.step(actions: actions, into: &result)
                observation = result.observations
                let manipulation = task.environment.manipulationStates()
                let clearances = result.metrics["state/box_clearance_m"]!
                let left = result.metrics["state/left_hand_contact"]!
                let right = result.metrics["state/right_hand_contact"]!
                let pedestal = result.metrics["state/box_pedestal_contact"]!
                let destination = result.metrics[
                    "state/box_destination_contact"]!
                let carryDistances = result.metrics["state/carry_distance_m"]!
                for environment in 0..<configuration.populationSize {
                    let rise = manipulation[environment].object.position.z
                        - initialManipulation[environment].object.position.z
                    maximumCenterRise[environment] = max(
                        maximumCenterRise[environment], rise)
                    maximumClearance[environment] = max(
                        maximumClearance[environment], clearances[environment])
                    let destinationTarget = task.currentPlacementTarget(
                        environment: environment)
                    let placementPlanarDistance = hypot(
                        manipulation[environment].object.position.x
                            - destinationTarget.x,
                        manipulation[environment].object.position.y
                            - destinationTarget.y)
                    positiveClearanceSums[environment] += max(
                        clearances[environment], 0)
                    let bilateral = left[environment] > 0.5
                        && right[environment] > 0.5
                    if bilateral { bilateralSteps[environment] += 1 }
                    let upright = manipulation[environment].object.rotation
                        .act(F3(0, 0, 1)).z
                    minimumUpright[environment] = min(
                        minimumUpright[environment], upright)
                    physicallyLifted[environment] = physicallyLifted[environment]
                        || observation.policy[
                            environment * HumanoidBoxCarryTask.observationDimension
                                + 89] > 0.5
                    if physicallyLifted[environment]
                        && destination[environment] > 0.5 {
                        destinationContactSteps[environment] += 1
                        minimumPlacementPlanarDistance[environment] = min(
                            minimumPlacementPlanarDistance[environment],
                            placementPlanarDistance)
                    }
                    let stableUnsupportedHold = physicallyLifted[environment]
                        && bilateral && pedestal[environment] < 0.5
                    if stableUnsupportedHold {
                        unsupportedSteps[environment] += 1
                        consecutiveStableUnsupportedSteps[environment] += 1
                        maximumStableUnsupportedSteps[environment] = max(
                            maximumStableUnsupportedSteps[environment],
                            consecutiveStableUnsupportedSteps[environment])
                        stablePositiveClearanceSums[environment] += max(
                            clearances[environment], 0)
                        maximumCarryDistance[environment] = max(
                            maximumCarryDistance[environment],
                            carryDistances[environment])
                        minimumPlacementPlanarDistance[environment] = min(
                            minimumPlacementPlanarDistance[environment],
                            placementPlanarDistance)
                    } else {
                        consecutiveStableUnsupportedSteps[environment] = 0
                    }
                    succeeded[environment] = succeeded[environment]
                        || result.successes[environment]
                    failed[environment] = failed[environment]
                        || result.truncated[environment]
                        || (result.terminated[environment]
                            && !result.successes[environment])
                }
            }

            let candidateResults = (0..<configuration.populationSize).map {
                environment in
                let bilateralFraction = Float(bilateralSteps[environment])
                    / Float(configuration.probeSteps)
                let unsupportedFraction = Float(unsupportedSteps[environment])
                    / Float(configuration.probeSteps)
                // The lowest oriented corner—not center rise—is the pickup
                // objective. The previous weights let a one-sided pivot gain
                // ~22 points from center rise while a real centimeter of air
                // gap was worth only 10. Make clearance dominant, reward its
                // duration, and retain center rise only as a weak tie-breaker.
                let meanPositiveClearance = positiveClearanceSums[environment]
                    / Float(configuration.probeSteps)
                let meanStablePositiveClearance =
                    stablePositiveClearanceSums[environment]
                        / Float(configuration.probeSteps)
                let maximumStableUnsupportedFraction =
                    Float(maximumStableUnsupportedSteps[environment])
                        / Float(configuration.probeSteps)
                let maximumDestinationProgress = max(
                    initialPlacementPlanarDistance
                        - minimumPlacementPlanarDistance[environment], 0)
                let destinationContactFraction = Float(
                    destinationContactSteps[environment])
                    / Float(configuration.probeSteps)
                var score: Float
                if configuration.sustainedCarryObjective {
                    // Feasibility is deliberately tiered. A non-lifting pivot
                    // must never outrank a real, if brief, unsupported pickup;
                    // once lifted, CEM is driven primarily by consecutive
                    // bilateral hold duration and then physical carry distance.
                    score = 200 * maximumClearance[environment]
                        + 2_000 * meanStablePositiveClearance
                        + 1_500 * unsupportedFraction
                        + 1_500 * maximumStableUnsupportedFraction
                        // Carry displacement alone rewards motion in any
                        // direction. Optimize measured progress toward the
                        // receiving table so CEM cannot prefer a robust walk
                        // that transports the box away from the task goal.
                        + 10_000 * maximumDestinationProgress
                        + 20_000 * destinationContactFraction
                        + 500 * maximumCarryDistance[environment]
                        + 3 * bilateralFraction
                        - 50 * max(1 - minimumUpright[environment], 0)
                        - (failed[environment] ? 300 : 0)
                } else {
                    score = 10_000 * maximumClearance[environment]
                        + 2_000 * meanPositiveClearance
                        + 25 * maximumCenterRise[environment]
                        + bilateralFraction
                        + 50 * unsupportedFraction
                        + 500 * maximumCarryDistance[environment]
                        - 20 * max(1 - minimumUpright[environment], 0)
                        - (failed[environment] ? 200 : 0)
                }
                if physicallyLifted[environment] {
                    score += configuration.sustainedCarryObjective ? 1_000 : 100
                }
                if succeeded[environment] {
                    score += configuration.sustainedCarryObjective ? 50_000 : 300
                }
                return CandidateResult(
                    score: score,
                    armTarget: candidates[environment],
                    maximumCenterRise: maximumCenterRise[environment],
                    maximumClearance: maximumClearance[environment],
                    bilateralFraction: bilateralFraction,
                    unsupportedFraction: unsupportedFraction,
                    minimumUpright: minimumUpright[environment],
                    physicallyLifted: physicallyLifted[environment],
                    failed: failed[environment],
                    maximumCarryDistance:
                        maximumCarryDistance[environment],
                    maximumStableUnsupportedSteps:
                        maximumStableUnsupportedSteps[environment],
                    minimumPlacementPlanarDistance:
                        minimumPlacementPlanarDistance[environment],
                    maximumDestinationProgress: maximumDestinationProgress,
                    destinationContactFraction: destinationContactFraction,
                    succeeded: succeeded[environment])
            }.sorted { lhs, rhs in
                isFeasibilityPreferred(
                    succeeded: lhs.succeeded,
                    physicallyLifted: lhs.physicallyLifted,
                    score: lhs.score,
                    overSucceeded: rhs.succeeded,
                    overPhysicallyLifted: rhs.physicallyLifted,
                    overScore: rhs.score)
            }
            let generationBest = candidateResults[0]
            if overallBest == nil || isFeasibilityPreferred(
                succeeded: generationBest.succeeded,
                physicallyLifted: generationBest.physicallyLifted,
                score: generationBest.score,
                overSucceeded: overallBest!.succeeded,
                overPhysicallyLifted: overallBest!.physicallyLifted,
                overScore: overallBest!.score) {
                overallBest = generationBest
                overallBestGeneration = generation
            }
            generationReports.append(.init(
                generation: generation,
                warmupSteps: warmupSteps,
                bestScore: generationBest.score,
                bestArmTarget: generationBest.armTarget,
                maximumBoxCenterRiseMeters: generationBest.maximumCenterRise,
                maximumBoxClearanceMeters: generationBest.maximumClearance,
                bilateralContactFraction: generationBest.bilateralFraction,
                unsupportedFraction: generationBest.unsupportedFraction,
                minimumBoxUprightAlignment: generationBest.minimumUpright,
                physicallyLifted: generationBest.physicallyLifted,
                failed: generationBest.failed,
                maximumCarryDistanceMeters:
                    generationBest.maximumCarryDistance,
                maximumStableUnsupportedSteps:
                    generationBest.maximumStableUnsupportedSteps,
                minimumPlacementPlanarDistanceMeters:
                    generationBest.minimumPlacementPlanarDistance,
                maximumDestinationProgressMeters:
                    generationBest.maximumDestinationProgress,
                destinationContactFraction:
                    generationBest.destinationContactFraction,
                succeeded: generationBest.succeeded))

            let eliteCount = max(
                2, Int(Float(configuration.populationSize)
                    * configuration.eliteFraction))
            let elites = candidateResults.prefix(eliteCount)
            var nextMean = [Float](repeating: 0, count: searchDimension)
            for elite in elites {
                for parameter in 0..<searchDimension {
                    nextMean[parameter] += elite.armTarget[parameter]
                }
            }
            for parameter in 0..<searchDimension {
                nextMean[parameter] /= Float(eliteCount)
            }
            var nextStandardDeviation = [Float](
                repeating: 0, count: searchDimension)
            for elite in elites {
                for parameter in 0..<searchDimension {
                    let delta = elite.armTarget[parameter]
                        - nextMean[parameter]
                    nextStandardDeviation[parameter] += delta * delta
                }
            }
            for parameter in 0..<searchDimension {
                nextStandardDeviation[parameter] = simd_clamp(
                    sqrt(nextStandardDeviation[parameter]
                        / Float(eliteCount)),
                    0.025, 0.60)
            }
            // Mild smoothing keeps one lucky contact transient from collapsing
            // the search distribution in a single generation.
            searchMean = zip(searchMean, nextMean).map {
                0.25 * $0.0 + 0.75 * $0.1
            }
            searchStandardDeviation = zip(
                searchStandardDeviation, nextStandardDeviation).map {
                    0.25 * $0.0 + 0.75 * $0.1
                }
        }

        let best = overallBest!
        return HumanoidBoxCarryActuationProbeReport(
            checkpointDirectory: checkpointDirectory,
            checkpointTaskRevision: runner.metadata.taskRevision ?? 1,
            populationSize: configuration.populationSize,
            generations: generationReports,
            bestGeneration: overallBestGeneration,
            bestScore: best.score,
            bestArmTarget: best.armTarget,
            maximumBoxCenterRiseMeters: best.maximumCenterRise,
            maximumBoxClearanceMeters: best.maximumClearance,
            physicallyLifted: best.physicallyLifted,
            maximumCarryDistanceMeters: best.maximumCarryDistance,
            maximumStableUnsupportedSteps:
                best.maximumStableUnsupportedSteps,
            minimumPlacementPlanarDistanceMeters:
                best.minimumPlacementPlanarDistance,
            maximumDestinationProgressMeters:
                best.maximumDestinationProgress,
            destinationContactFraction: best.destinationContactFraction,
            succeeded: best.succeeded)
    }

    /// Feasibility tiers are lexicographic: a task success outranks every
    /// failure, and a measured unsupported lift outranks every supported
    /// pivot. The shaped score optimizes only within the same physical tier.
    /// This keeps large but invalid clearance transients out of CEM's elite
    /// set as well as out of the final saved trajectory.
    static func isFeasibilityPreferred(
        succeeded: Bool, physicallyLifted: Bool, score: Float,
        overSucceeded: Bool, overPhysicallyLifted: Bool, overScore: Float
    ) -> Bool {
        if succeeded != overSucceeded { return succeeded }
        if physicallyLifted != overPhysicallyLifted { return physicallyLifted }
        return score > overScore
    }

    private static func policyActions(
        runner: VectorPolicyRunner,
        task: HumanoidBoxCarryTask,
        observation: RLObservationBatch
    ) throws -> RLActionBatch {
        try runner.actions(
            for: observation,
            expertGates: task.policyExpertGates(observation.policy),
            expertActionMask: task.policyExpertActionMask,
            standExpertGates: task.policyStandExpertGates(observation.policy),
            standExpertActionMask: task.policyStandExpertActionMask,
            auxiliaryExpertGates:
                task.policyAuxiliaryExpertGates(observation.policy),
            auxiliaryExpertActionMask:
                task.policyAuxiliaryExpertActionMask)
    }

    static func sampledTargets(
        count: Int,
        mean: [Float],
        standardDeviation: [Float],
        generator: inout ProbeRandomNumberGenerator
    ) -> [[Float]] {
        precondition(count > 0 && !mean.isEmpty
            && standardDeviation.count == mean.count)
        var result = [[Float]]()
        result.reserveCapacity(count)
        // Always retain the current mean, then sample antithetic pairs. This
        // makes every generation no worse than its center under deterministic
        // physics and substantially reduces variance for a small population.
        result.append(mean.map { simd_clamp($0, -0.999, 0.999) })
        while result.count < count {
            let z = mean.map { _ in generator.normal() }
            let positive = zip(zip(mean, standardDeviation), z).map {
                simd_clamp($0.0.0 + $0.0.1 * $0.1, -0.999, 0.999)
            }
            result.append(positive)
            if result.count < count {
                let negative = zip(zip(mean, standardDeviation), z).map {
                    simd_clamp($0.0.0 - $0.0.1 * $0.1, -0.999, 0.999)
                }
                result.append(negative)
            }
        }
        return result
    }

    /// Smooth symmetric post-contact action delta. Knot zero is fixed at zero
    /// so the optimizer cannot introduce an impulse at the grasp boundary;
    /// the searched knots are linearly interpolated over the probe horizon.
    static func trajectoryArmDelta(
        _ parameters: [Float], knotCount: Int, progress: Float
    ) -> [Float] {
        precondition(knotCount > 0 && parameters.count == 4 * knotCount)
        let scaled = simd_clamp(progress, 0, 1) * Float(knotCount)
        let lower = min(Int(floor(scaled)), knotCount - 1)
        let fraction = scaled - Float(lower)
        func knot(_ index: Int) -> [Float] {
            guard index > 0 else { return [Float](repeating: 0, count: 4) }
            let base = (index - 1) * 4
            return Array(parameters[base..<(base + 4)])
        }
        let a = knot(lower)
        let b = knot(min(lower + 1, knotCount))
        let unique = zip(a, b).map {
            (1 - fraction) * $0.0 + fraction * $0.1
        }
        return [
            unique[0], unique[1], unique[2], unique[3],
            unique[0], -unique[1], -unique[2], unique[3],
        ]
    }
}

struct ProbeRandomNumberGenerator {
    private var state: UInt64
    private var spareNormal: Float?

    init(seed: UInt64) {
        state = seed &+ 0x9E3779B97F4A7C15
    }

    mutating func nextUnit() -> Float {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z ^= z >> 31
        return Float(Double(z >> 11) * (1.0 / 9_007_199_254_740_992.0))
    }

    mutating func normal() -> Float {
        if let spareNormal {
            self.spareNormal = nil
            return spareNormal
        }
        let u1 = max(nextUnit(), 1e-7)
        let u2 = nextUnit()
        let radius = sqrt(-2 * log(u1))
        let angle = 2 * Float.pi * u2
        spareNormal = radius * sin(angle)
        return radius * cos(angle)
    }
}
