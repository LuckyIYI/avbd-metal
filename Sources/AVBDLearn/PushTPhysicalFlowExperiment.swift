import AVBDCore
import Foundation
import simd

/// Endpoint-clamped, open-uniform cubic B-spline utilities. Keeping this
/// simulator-side implementation independent of MLX avoids hidden graph or
/// synchronization costs in exact-physics planning.
public enum PushTClampedCubicSpline {
    public static func sample(
        controlPoints: [SIMD2<Float>], progress: Float
    ) -> SIMD2<Float> {
        precondition(controlPoints.count >= 4,
                     "a cubic B-spline requires four control points")
        let weights = basisWeights(
            controlPointCount: controlPoints.count, progress: progress)
        return zip(controlPoints, weights).reduce(.zero) {
            $0 + $1.0 * $1.1
        }
    }

    /// Fits the action sequence while fixing both control endpoints to the
    /// source and target actuator states. This diagnostic is never a seed for
    /// the state-only optimizer.
    public static func fit(
        samples: [SIMD2<Float>],
        initialControlPoint: SIMD2<Float>,
        controlPointCount: Int,
        ridge: Float = 1e-5
    ) -> [SIMD2<Float>] {
        precondition(samples.count >= 2)
        precondition(controlPointCount >= 4)
        let unknownCount = controlPointCount - 2
        let finalControlPoint = samples.last!
        var normal = [[Float]](
            repeating: [Float](repeating: 0, count: unknownCount),
            count: unknownCount)
        var rhsX = [Float](repeating: 0, count: unknownCount)
        var rhsY = [Float](repeating: 0, count: unknownCount)
        for sampleIndex in samples.indices {
            let progress = Float(sampleIndex) / Float(samples.count - 1)
            let weights = basisWeights(
                controlPointCount: controlPointCount, progress: progress)
            let adjusted = samples[sampleIndex]
                - initialControlPoint * weights[0]
                - finalControlPoint * weights[controlPointCount - 1]
            for row in 0..<unknownCount {
                let coefficient = weights[row + 1]
                rhsX[row] += coefficient * adjusted.x
                rhsY[row] += coefficient * adjusted.y
                for column in 0..<unknownCount {
                    normal[row][column] += coefficient * weights[column + 1]
                }
            }
        }
        for index in 0..<unknownCount { normal[index][index] += ridge }
        let x = solve(normal, rhsX)
        let y = solve(normal, rhsY)
        return [initialControlPoint] + (0..<unknownCount).map {
            simd_clamp(SIMD2(x[$0], y[$0]), SIMD2(repeating: -3),
                       SIMD2(repeating: 3))
        } + [finalControlPoint]
    }

    static func basisWeights(
        controlPointCount: Int, progress: Float
    ) -> [Float] {
        precondition(controlPointCount >= 4)
        let degree = 3
        let lastControlPoint = controlPointCount - 1
        let u = simd_clamp(progress, 0, 1)
        if u >= 1 {
            var endpoint = [Float](repeating: 0, count: controlPointCount)
            endpoint[lastControlPoint] = 1
            return endpoint
        }
        let knotCount = lastControlPoint + degree + 2
        let internalDenominator = Float(lastControlPoint - degree + 1)
        let knots = (0..<knotCount).map { index -> Float in
            if index <= degree { return 0 }
            if index >= lastControlPoint + 1 { return 1 }
            return Float(index - degree) / internalDenominator
        }
        var basis = [Float](repeating: 0, count: controlPointCount + 1)
        for index in 0..<controlPointCount
            where knots[index] <= u && u < knots[index + 1] {
            basis[index] = 1
        }
        for order in 1...degree {
            var next = [Float](repeating: 0, count: controlPointCount + 1)
            for index in 0..<controlPointCount {
                let leftDenominator = knots[index + order] - knots[index]
                if leftDenominator > 0 {
                    next[index] += (u - knots[index]) / leftDenominator
                        * basis[index]
                }
                let rightDenominator = knots[index + order + 1]
                    - knots[index + 1]
                if rightDenominator > 0 {
                    next[index] += (knots[index + order + 1] - u)
                        / rightDenominator * basis[index + 1]
                }
            }
            basis = next
        }
        return Array(basis.prefix(controlPointCount))
    }

    private static func solve(_ matrix: [[Float]], _ rhs: [Float])
        -> [Float] {
        var a = matrix
        var b = rhs
        let count = rhs.count
        for column in 0..<count {
            var pivot = column
            for row in (column + 1)..<count
                where abs(a[row][column]) > abs(a[pivot][column]) {
                pivot = row
            }
            precondition(abs(a[pivot][column]) > 1e-10,
                         "regularized spline fit became singular")
            if pivot != column {
                a.swapAt(pivot, column)
                b.swapAt(pivot, column)
            }
            let divisor = a[column][column]
            for index in column..<count { a[column][index] /= divisor }
            b[column] /= divisor
            for row in 0..<count where row != column {
                let factor = a[row][column]
                guard factor != 0 else { continue }
                for index in column..<count {
                    a[row][index] -= factor * a[column][index]
                }
                b[row] -= factor * b[column]
            }
        }
        return b
    }
}

public enum PushTPhysicalFlowProposal: String, Codable, Sendable {
    case linearEndpoints
    case endpointContact
}

public enum PushTCEMCovarianceMode: String, Codable, Sendable {
    case diagonal
    case full
}

public struct PushTPhysicalFlowConfiguration: Sendable {
    public var populationSize: Int
    public var generations: Int
    public var horizon: Int
    public var controlPointCount: Int
    public var substeps: Int
    public var sourcePreparationSteps: Int
    public var targetSettlingSteps: Int
    public var terminalHoldSteps: Int
    public var targetGenerationMaximumStep: Float
    public var proposal: PushTPhysicalFlowProposal
    public var covarianceMode: PushTCEMCovarianceMode
    public var initialStandardDeviation: Float
    public var eliteFraction: Float
    public var seed: UInt64

    public init(
        populationSize: Int = 256,
        generations: Int = 6,
        horizon: Int = 48,
        controlPointCount: Int = 6,
        substeps: Int = 4,
        sourcePreparationSteps: Int = 160,
        targetSettlingSteps: Int = 0,
        terminalHoldSteps: Int = 0,
        targetGenerationMaximumStep: Float = 0.16,
        proposal: PushTPhysicalFlowProposal = .linearEndpoints,
        covarianceMode: PushTCEMCovarianceMode = .diagonal,
        initialStandardDeviation: Float = 0.65,
        eliteFraction: Float = 0.1,
        seed: UInt64 = 1
    ) {
        self.populationSize = populationSize
        self.generations = generations
        self.horizon = horizon
        self.controlPointCount = controlPointCount
        self.substeps = substeps
        self.sourcePreparationSteps = sourcePreparationSteps
        self.targetSettlingSteps = targetSettlingSteps
        self.terminalHoldSteps = terminalHoldSteps
        self.targetGenerationMaximumStep = targetGenerationMaximumStep
        self.proposal = proposal
        self.covarianceMode = covarianceMode
        self.initialStandardDeviation = initialStandardDeviation
        self.eliteFraction = eliteFraction
        self.seed = seed
    }

    func validate() throws {
        guard populationSize >= 8,
              generations > 0,
              horizon >= 8,
              controlPointCount >= 4,
              controlPointCount <= horizon + 1,
              substeps > 0,
              sourcePreparationSteps > 0,
              targetSettlingSteps >= 0,
              targetSettlingSteps < horizon,
              terminalHoldSteps >= 0,
              terminalHoldSteps < horizon,
              targetGenerationMaximumStep.isFinite,
              targetGenerationMaximumStep > 0,
              targetGenerationMaximumStep <= 0.16,
              initialStandardDeviation.isFinite,
              initialStandardDeviation > 0,
              eliteFraction > 0,
              eliteFraction <= 0.5 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Push-T physical-flow experiment configuration")
        }
    }
}

public struct PushTPhysicalFlowMetrics: Codable, Sendable {
    public var loss: Float
    public var blockPositionErrorMeters: Float
    public var blockYawErrorRadians: Float
    public var blockLinearVelocityErrorMPS: Float
    public var blockAngularVelocityErrorRadPS: Float
    public var tipPositionErrorMeters: Float
    public var tipLinearVelocityErrorMPS: Float
    public var commandedTargetErrorMeters: Float
}

public struct PushTPhysicalFlowGeneration: Codable, Sendable {
    public var generation: Int
    public var bestLoss: Float
    public var medianLoss: Float
    public var distributionMeanLoss: Float
    public var meanStandardDeviationMeters: Float
}

public struct PushTPhysicalFlowReport: Codable, Sendable {
    public var experiment: String
    public var seed: UInt64
    public var populationSize: Int
    public var generations: Int
    public var horizon: Int
    public var controlPointCount: Int
    public var substeps: Int
    public var sourcePreparationSteps: Int
    public var targetSettlingSteps: Int
    public var terminalHoldSteps: Int
    public var targetGenerationMaximumStep: Float
    public var proposal: PushTPhysicalFlowProposal
    public var covarianceMode: PushTCEMCovarianceMode
    public var sourceHadActiveContact: Bool
    public var targetBlockDisplacementMeters: Float
    public var targetBlockYawChangeRadians: Float
    public var targetBlockLinearSpeedMPS: Float
    public var targetBlockAngularSpeedRadPS: Float
    public var targetBlockCenterHeightMeters: Float
    public var targetBlockVerticalSpeedMPS: Float
    public var targetBlockGroundContact: Bool
    public var targetTipBlockContact: Bool
    public var targetCloneSpreadLoss: Float
    public var targetCloneMaximumStateError: Float
    public var referenceReplay: PushTPhysicalFlowMetrics
    public var linearInterpolation: PushTPhysicalFlowMetrics
    public var fittedReferenceSpline: PushTPhysicalFlowMetrics
    public var initialProposalSpline: PushTPhysicalFlowMetrics
    public var randomShooting: PushTPhysicalFlowMetrics
    public var optimizedSpline: PushTPhysicalFlowMetrics
    public var optimizedToLinearLossRatio: Float
    public var optimizedToRandomLossRatio: Float
    public var selectedReplayLoss: Float
    public var selectedReplayMaximumStateError: Float
    public var bestControlPointsXY: [Float]
    public var generationHistory: [PushTPhysicalFlowGeneration]
    public var candidateRollouts: Int
    public var simulatedEnvironmentControlSteps: Int
    public var elapsedSeconds: Double
    public var infrastructureGatePassed: Bool
    public var poseReconstructionGatePassed: Bool
    public var dynamicStateReconstructionGatePassed: Bool
    public var goGatePassed: Bool
}

/// Given two states produced by AVBD, search for a smooth, physically executed
/// flow from the first to the second. Generating actions are kept away from CEM
/// and used only for an explicitly labeled representation/replay diagnostic.
public enum PushTPhysicalFlowExperiment {
    private struct Candidate {
        var environment: Int
        var controlPoints: [SIMD2<Float>]
        var metrics: PushTPhysicalFlowMetrics
        var terminal: PushTPhysicalState
    }

    public static func run(
        configuration: PushTPhysicalFlowConfiguration = .init()
    ) throws -> PushTPhysicalFlowReport {
        try configuration.validate()
        let startTime = Date()
        let population = configuration.populationSize
        let environment = try PushTEnv(
            numEnvs: population, seed: configuration.seed,
            layout: .coLocated)
        environment.resetAll(seed: configuration.seed &+ 17)

        var sourcePreparationSteps = 0
        var sourceHasContact = false
        while sourcePreparationSteps < configuration.sourcePreparationSteps,
              !sourceHasContact {
            let action = environment.oracleAction(0)
            environment.step(
                actions: [SIMD2<Float>](repeating: action, count: population),
                substeps: configuration.substeps,
                maxStep: configuration.targetGenerationMaximumStep)
            sourcePreparationSteps += 1
            sourceHasContact = hasTipBlockContact(environment, environment: 0)
        }
        guard sourceHasContact else {
            throw RLEnvironmentError.invalidConfiguration(
                "Push-T oracle failed to establish source contact")
        }
        let source = environment.physicalStates()[0]
        environment.fork(source, into: Array(0..<population))

        var referenceActions = [SIMD2<Float>]()
        referenceActions.reserveCapacity(configuration.horizon)
        let oracleSteps = configuration.horizon
            - configuration.targetSettlingSteps
        var fixedSettlingTarget: SIMD2<Float>?
        for targetStep in 0..<configuration.horizon {
            let action: SIMD2<Float>
            if targetStep < oracleSteps {
                action = environment.oracleAction(0)
            } else {
                // Select one retraction target and hold it. Recomputing
                // `tip + offset` each frame would march to the arena wall and
                // keep injecting energy rather than creating a settled state.
                if fixedSettlingTarget == nil {
                    let state = environment.states()[0]
                    let away = state.tipPosition - state.blockPosition
                    let direction = length(away) > 1e-4
                        ? normalize(away) : SIMD2<Float>(1, 0)
                    fixedSettlingTarget = simd_clamp(
                        state.tipPosition + direction * 0.6,
                        SIMD2(repeating: -3), SIMD2(repeating: 3))
                }
                action = fixedSettlingTarget!
            }
            referenceActions.append(action)
            environment.step(
                actions: [SIMD2<Float>](repeating: action, count: population),
                substeps: configuration.substeps, maxStep: 0.16)
        }
        let targetReplicas = environment.physicalStates()
        let target = targetReplicas[0]
        let targetContactPairs = environment.solver.activeRigidContactPairs()
        let targetReference = environment.refs[0]
        let targetBlockGroundContact = targetContactPairs.contains { pair in
            (pair.0 == environment.groundBody
                && (pair.1 == targetReference.blockBar
                    || pair.1 == targetReference.blockStem))
                || (pair.1 == environment.groundBody
                    && (pair.0 == targetReference.blockBar
                        || pair.0 == targetReference.blockStem))
        }
        let targetTipBlockContact = targetContactPairs.contains { pair in
            (pair.0 == targetReference.tip
                && (pair.1 == targetReference.blockBar
                    || pair.1 == targetReference.blockStem))
                || (pair.1 == targetReference.tip
                    && (pair.0 == targetReference.blockBar
                        || pair.0 == targetReference.blockStem))
        }
        let targetCloneSpread = targetReplicas.dropFirst().map {
            metrics(from: $0, to: target).loss
        }.max() ?? 0
        let targetCloneMaximumError = targetReplicas.dropFirst().map {
            maximumStateError($0, target)
        }.max() ?? 0

        let motionSteps = configuration.horizon
            - configuration.terminalHoldSteps
        let linearPath = (0..<configuration.horizon).map { step in
            let progress = min(
                Float(step + 1) / Float(motionSteps), 1)
            return simd_mix(source.commandedTipTarget,
                            target.commandedTipTarget,
                            SIMD2(repeating: progress))
        }
        let fittedControlPoints = PushTClampedCubicSpline.fit(
            samples: [source.commandedTipTarget] + referenceActions,
            initialControlPoint: source.commandedTipTarget,
            controlPointCount: configuration.controlPointCount)
        let fittedPath = path(
            controlPoints: fittedControlPoints, horizon: configuration.horizon,
            terminalHoldSteps: configuration.terminalHoldSteps)
        var calibrationPaths = [[SIMD2<Float>]](
            repeating: linearPath, count: population)
        // Index zero generated the canonical target, so replay the reference
        // there. Cross-replica agreement remains a separate reported guard.
        calibrationPaths[0] = referenceActions
        calibrationPaths[2] = fittedPath
        let calibrationStates = rollout(
            environment: environment, source: source,
            paths: calibrationPaths, configuration: configuration)
        let referenceMetrics = metrics(from: calibrationStates[0], to: target)
        let linearMetrics = metrics(from: calibrationStates[1], to: target)
        let fittedMetrics = metrics(from: calibrationStates[2], to: target)

        var generator = PhysicalFlowRandomNumberGenerator(
            seed: configuration.seed &+ 0xA11CE)
        let initialProposal = proposalControlPoints(
            source: source, target: target,
            count: configuration.controlPointCount,
            proposal: configuration.proposal)
        var searchMean = Array(
            initialProposal.dropFirst().dropLast()).flatMap {
            [$0.x, $0.y]
        }
        var searchCovariance = [[Float]](
            repeating: [Float](repeating: 0, count: searchMean.count),
            count: searchMean.count)
        for index in searchMean.indices {
            searchCovariance[index][index] =
                configuration.initialStandardDeviation
                    * configuration.initialStandardDeviation
        }
        var overallBest: Candidate?
        var randomShootingBest: Candidate?
        var initialProposalMetrics: PushTPhysicalFlowMetrics?
        var generationHistory = [PushTPhysicalFlowGeneration]()

        for generation in 0..<configuration.generations {
            let samplingTransform = cholesky(searchCovariance)
            var controls = [[SIMD2<Float>]]()
            controls.reserveCapacity(population)
            for candidateIndex in 0..<population {
                let parameters: [Float]
                if candidateIndex == 0 {
                    parameters = searchMean
                } else if candidateIndex == 1, let overallBest {
                    parameters = Array(overallBest.controlPoints
                        .dropFirst().dropLast())
                        .flatMap { [$0.x, $0.y] }
                } else {
                    let noise = searchMean.map { _ in generator.normal() }
                    parameters = searchMean.indices.map { row in
                        let delta = (0...row).reduce(Float(0)) {
                            $0 + samplingTransform[row][$1] * noise[$1]
                        }
                        return simd_clamp(searchMean[row] + delta, -3, 3)
                    }
                }
                controls.append(
                    [source.commandedTipTarget] + stride(
                        from: 0, to: parameters.count, by: 2).map {
                            SIMD2(parameters[$0], parameters[$0 + 1])
                        } + [target.commandedTipTarget])
            }
            let paths = controls.map {
                path(controlPoints: $0, horizon: configuration.horizon,
                     terminalHoldSteps: configuration.terminalHoldSteps)
            }
            let terminals = rollout(
                environment: environment, source: source,
                paths: paths, configuration: configuration)
            let candidates = (0..<population).map { index in
                Candidate(environment: index,
                          controlPoints: controls[index],
                          metrics: metrics(from: terminals[index], to: target),
                          terminal: terminals[index])
            }
            guard candidates.allSatisfy({ $0.metrics.loss.isFinite }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite Push-T physical-flow candidate loss")
            }
            let sorted = candidates.sorted { $0.metrics.loss < $1.metrics.loss }
            if overallBest == nil
                || sorted[0].metrics.loss < overallBest!.metrics.loss {
                overallBest = sorted[0]
            }
            if generation == 0 {
                initialProposalMetrics = candidates[0].metrics
                randomShootingBest = candidates[1...].min {
                    $0.metrics.loss < $1.metrics.loss
                }
            }
            let eliteCount = max(
                2, Int(Float(population) * configuration.eliteFraction))
            let elites = sorted.prefix(eliteCount)
            var nextMean = [Float](repeating: 0, count: searchMean.count)
            for elite in elites {
                let values = Array(elite.controlPoints
                    .dropFirst().dropLast()).flatMap {
                    [$0.x, $0.y]
                }
                for index in nextMean.indices { nextMean[index] += values[index] }
            }
            for index in nextMean.indices { nextMean[index] /= Float(eliteCount) }
            var nextCovariance = [[Float]](
                repeating: [Float](repeating: 0, count: searchMean.count),
                count: searchMean.count)
            for elite in elites {
                let values = Array(elite.controlPoints
                    .dropFirst().dropLast()).flatMap {
                    [$0.x, $0.y]
                }
                for row in searchMean.indices {
                    let rowDelta = values[row] - nextMean[row]
                    for column in searchMean.indices {
                        nextCovariance[row][column] += rowDelta
                            * (values[column] - nextMean[column])
                    }
                }
            }
            for row in searchMean.indices {
                for column in searchMean.indices {
                    nextCovariance[row][column] /= Float(eliteCount)
                    if configuration.covarianceMode == .diagonal,
                       row != column {
                        nextCovariance[row][column] = 0
                    }
                }
            }
            // Preserve covariance while bounding every marginal standard
            // deviation. Scaling rows and columns together preserves PSD.
            var marginalScale = [Float](repeating: 1, count: searchMean.count)
            for index in searchMean.indices {
                let deviation = sqrt(max(nextCovariance[index][index], 1e-12))
                if deviation > 1.2 { marginalScale[index] = 1.2 / deviation }
            }
            for row in searchMean.indices {
                for column in searchMean.indices {
                    nextCovariance[row][column] *= marginalScale[row]
                        * marginalScale[column]
                }
                nextCovariance[row][row] = max(
                    nextCovariance[row][row], 0.025 * 0.025)
            }
            searchMean = zip(searchMean, nextMean).map {
                0.25 * $0.0 + 0.75 * $0.1
            }
            for row in searchMean.indices {
                for column in searchMean.indices {
                    searchCovariance[row][column] =
                        0.25 * searchCovariance[row][column]
                        + 0.75 * nextCovariance[row][column]
                }
            }
            let losses = candidates.map(\.metrics.loss).sorted()
            generationHistory.append(.init(
                generation: generation,
                bestLoss: sorted[0].metrics.loss,
                medianLoss: losses[losses.count / 2],
                distributionMeanLoss: candidates[0].metrics.loss,
                meanStandardDeviationMeters:
                    searchMean.indices.reduce(0) {
                        $0 + sqrt(max(searchCovariance[$1][$1], 0))
                    } / Float(searchMean.count)))
        }

        let best = overallBest!
        let random = randomShootingBest!
        let bestPath = path(
            controlPoints: best.controlPoints, horizon: configuration.horizon,
            terminalHoldSteps: configuration.terminalHoldSteps)
        let replayStates = rollout(
            environment: environment, source: source,
            paths: [[SIMD2<Float>]](repeating: bestPath, count: population),
            configuration: configuration)
        let selectedReplayState = replayStates[best.environment]
        let selectedReplay = metrics(
            from: selectedReplayState, to: best.terminal)
        let selectedReplayMaximumError = maximumStateError(
            selectedReplayState, best.terminal)
        let sourceBlockCenter = blockCenter(source)
        let targetBlockCenter = blockCenter(target)
        let targetDisplacement = length(SIMD2(
            targetBlockCenter.x - sourceBlockCenter.x,
            targetBlockCenter.y - sourceBlockCenter.y))
        let targetYawChange = abs(wrappedAngle(blockYaw(target) - blockYaw(source)))
        let targetLinearVelocity = 0.5 * (
            target.blockBar.linearVelocity + target.blockStem.linearVelocity)
        let targetAngularVelocity = 0.5 * (
            target.blockBar.angularVelocity + target.blockStem.angularVelocity)
        let optimizedToLinear = best.metrics.loss / max(linearMetrics.loss, 1e-12)
        let optimizedToRandom = best.metrics.loss / max(random.metrics.loss, 1e-12)
        let infrastructureGatePassed = targetCloneMaximumError < 0.005
            && referenceMetrics.loss < 1e-7
            && selectedReplayMaximumError < 1e-4
        let poseGatePassed = targetDisplacement >= 0.05
            && best.metrics.blockPositionErrorMeters < 0.05
            && best.metrics.blockYawErrorRadians < 0.1
            && optimizedToLinear < 0.8
            && optimizedToRandom < 0.8
        let dynamicGatePassed = poseGatePassed
            && best.metrics.blockLinearVelocityErrorMPS < 0.1
            && best.metrics.blockAngularVelocityErrorRadPS < 0.25
            && best.metrics.tipPositionErrorMeters < 0.1
            && best.metrics.commandedTargetErrorMeters < 0.1
        let batchedPhysicalFlows = configuration.generations + 3
        return PushTPhysicalFlowReport(
            experiment: "pusht-state-to-state-physical-flow-v0",
            seed: configuration.seed,
            populationSize: population,
            generations: configuration.generations,
            horizon: configuration.horizon,
            controlPointCount: configuration.controlPointCount,
            substeps: configuration.substeps,
            sourcePreparationSteps: sourcePreparationSteps,
            targetSettlingSteps: configuration.targetSettlingSteps,
            terminalHoldSteps: configuration.terminalHoldSteps,
            targetGenerationMaximumStep:
                configuration.targetGenerationMaximumStep,
            proposal: configuration.proposal,
            covarianceMode: configuration.covarianceMode,
            sourceHadActiveContact: sourceHasContact,
            targetBlockDisplacementMeters: targetDisplacement,
            targetBlockYawChangeRadians: targetYawChange,
            targetBlockLinearSpeedMPS: length(targetLinearVelocity),
            targetBlockAngularSpeedRadPS: length(targetAngularVelocity),
            targetBlockCenterHeightMeters: targetBlockCenter.z,
            targetBlockVerticalSpeedMPS: targetLinearVelocity.z,
            targetBlockGroundContact: targetBlockGroundContact,
            targetTipBlockContact: targetTipBlockContact,
            targetCloneSpreadLoss: targetCloneSpread,
            targetCloneMaximumStateError: targetCloneMaximumError,
            referenceReplay: referenceMetrics,
            linearInterpolation: linearMetrics,
            fittedReferenceSpline: fittedMetrics,
            initialProposalSpline: initialProposalMetrics!,
            randomShooting: random.metrics,
            optimizedSpline: best.metrics,
            optimizedToLinearLossRatio: optimizedToLinear,
            optimizedToRandomLossRatio: optimizedToRandom,
            selectedReplayLoss: selectedReplay.loss,
            selectedReplayMaximumStateError: selectedReplayMaximumError,
            bestControlPointsXY: best.controlPoints.flatMap { [$0.x, $0.y] },
            generationHistory: generationHistory,
            candidateRollouts: population * (configuration.generations + 2),
            simulatedEnvironmentControlSteps: population * (
                sourcePreparationSteps + configuration.horizon
                    * batchedPhysicalFlows),
            elapsedSeconds: Date().timeIntervalSince(startTime),
            infrastructureGatePassed: infrastructureGatePassed,
            poseReconstructionGatePassed: poseGatePassed,
            dynamicStateReconstructionGatePassed: dynamicGatePassed,
            goGatePassed: infrastructureGatePassed && dynamicGatePassed)
    }

    private static func rollout(
        environment: PushTEnv,
        source: PushTPhysicalState,
        paths: [[SIMD2<Float>]],
        configuration: PushTPhysicalFlowConfiguration
    ) -> [PushTPhysicalState] {
        precondition(paths.count == configuration.populationSize)
        precondition(paths.allSatisfy { $0.count == configuration.horizon })
        environment.fork(source, into: Array(0..<configuration.populationSize))
        for step in 0..<configuration.horizon {
            environment.step(
                actions: paths.map { $0[step] },
                substeps: configuration.substeps, maxStep: 0.16)
        }
        return environment.physicalStates()
    }

    private static func path(
        controlPoints: [SIMD2<Float>], horizon: Int,
        terminalHoldSteps: Int
    ) -> [SIMD2<Float>] {
        let motionSteps = horizon - terminalHoldSteps
        return (0..<horizon).map { step in
            PushTClampedCubicSpline.sample(
                controlPoints: controlPoints,
                progress: min(Float(step + 1) / Float(motionSteps), 1))
        }
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

    /// State-only imagination proposal. It does not use the reference action
    /// sequence: the object endpoint displacement supplies a contact normal,
    /// then the final control point honors the target actuator state. CEM and
    /// exact physics remain responsible for contact side, rotation, and timing.
    private static func proposalControlPoints(
        source: PushTPhysicalState, target: PushTPhysicalState,
        count: Int, proposal: PushTPhysicalFlowProposal
    ) -> [SIMD2<Float>] {
        let sourceCommand = source.commandedTipTarget
        let targetCommand = target.commandedTipTarget
        guard proposal == .endpointContact else {
            return (0..<count).map { index in
                let progress = Float(index) / Float(count - 1)
                return simd_mix(sourceCommand, targetCommand,
                                SIMD2(repeating: progress))
            }
        }
        let sourceCenter3 = blockCenter(source)
        let targetCenter3 = blockCenter(target)
        let sourceCenter = SIMD2(sourceCenter3.x, sourceCenter3.y)
        let targetCenter = SIMD2(targetCenter3.x, targetCenter3.y)
        let displacement = targetCenter - sourceCenter
        let direction = length(displacement) > 1e-4
            ? normalize(displacement) : SIMD2<Float>(1, 0)
        let behind = sourceCenter - direction * 0.48
        let pushThrough = targetCenter + direction * 0.22
        var points = [sourceCommand, behind]
        let middleCount = count - 4
        if middleCount > 0 {
            for index in 1...middleCount {
                let progress = Float(index) / Float(middleCount + 1)
                points.append(simd_mix(
                    sourceCenter, targetCenter,
                    SIMD2(repeating: progress)))
            }
        }
        points.append(pushThrough)
        points.append(targetCommand)
        precondition(points.count == count)
        return points.map {
            simd_clamp($0, SIMD2(repeating: -3), SIMD2(repeating: 3))
        }
    }

    private static func metrics(
        from state: PushTPhysicalState, to target: PushTPhysicalState
    ) -> PushTPhysicalFlowMetrics {
        let blockPosition = length(blockCenter(state) - blockCenter(target))
        let blockYawError = abs(wrappedAngle(blockYaw(state) - blockYaw(target)))
        let stateLinearVelocity = 0.5 * (
            state.blockBar.linearVelocity + state.blockStem.linearVelocity)
        let targetLinearVelocity = 0.5 * (
            target.blockBar.linearVelocity + target.blockStem.linearVelocity)
        let blockLinearVelocity = length(
            stateLinearVelocity - targetLinearVelocity)
        let stateAngularVelocity = 0.5 * (
            state.blockBar.angularVelocity + state.blockStem.angularVelocity)
        let targetAngularVelocity = 0.5 * (
            target.blockBar.angularVelocity + target.blockStem.angularVelocity)
        let blockAngularVelocity = length(
            stateAngularVelocity - targetAngularVelocity)
        let tipPosition = length(state.tip.position - target.tip.position)
        let tipLinearVelocity = length(
            state.tip.linearVelocity - target.tip.linearVelocity)
        let command = length(
            state.commandedTipTarget - target.commandedTipTarget)
        let loss = 100 * blockPosition * blockPosition
            + 4 * blockYawError * blockYawError
            + 0.5 * blockLinearVelocity * blockLinearVelocity
            + 0.1 * blockAngularVelocity * blockAngularVelocity
            + 10 * tipPosition * tipPosition
            + 0.2 * tipLinearVelocity * tipLinearVelocity
            + 2 * command * command
        return PushTPhysicalFlowMetrics(
            loss: loss,
            blockPositionErrorMeters: blockPosition,
            blockYawErrorRadians: blockYawError,
            blockLinearVelocityErrorMPS: blockLinearVelocity,
            blockAngularVelocityErrorRadPS: blockAngularVelocity,
            tipPositionErrorMeters: tipPosition,
            tipLinearVelocityErrorMPS: tipLinearVelocity,
            commandedTargetErrorMeters: command)
    }

    private static func blockCenter(_ state: PushTPhysicalState) -> F3 {
        0.5 * (state.blockBar.position + state.blockStem.position)
    }

    private static func blockYaw(_ state: PushTPhysicalState) -> Float {
        let forward = state.blockBar.rotation.act(F3(1, 0, 0))
        return atan2(forward.y, forward.x)
    }

    private static func wrappedAngle(_ angle: Float) -> Float {
        angle - 2 * .pi * (angle / (2 * .pi)).rounded()
    }

    private static func hasTipBlockContact(
        _ environment: PushTEnv, environment index: Int
    ) -> Bool {
        let reference = environment.refs[index]
        return environment.solver.activeRigidContactPairs().contains { pair in
            (pair.0 == reference.tip
                && (pair.1 == reference.blockBar
                    || pair.1 == reference.blockStem))
                || (pair.1 == reference.tip
                    && (pair.0 == reference.blockBar
                        || pair.0 == reference.blockStem))
        }
    }

    private static func maximumStateError(
        _ lhs: PushTPhysicalState, _ rhs: PushTPhysicalState
    ) -> Float {
        func vector(_ a: F3, _ b: F3) -> Float {
            max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z))
        }
        func quaternion(_ a: Quat, _ b: Quat) -> Float {
            let sign: Float = simd_dot(a.vector, b.vector) < 0 ? -1 : 1
            return max(abs(a.real - sign * b.real),
                       vector(a.imag, sign * b.imag))
        }
        func rigid(
            _ a: GPUSolver.RigidBodyState, _ b: GPUSolver.RigidBodyState
        ) -> Float {
            [vector(a.position, b.position), quaternion(a.rotation, b.rotation),
             vector(a.linearVelocity, b.linearVelocity),
             vector(a.angularVelocity, b.angularVelocity)].max()!
        }
        return [rigid(lhs.tip, rhs.tip),
                rigid(lhs.blockBar, rhs.blockBar),
                rigid(lhs.blockStem, rhs.blockStem),
                abs(lhs.commandedTipTarget.x - rhs.commandedTipTarget.x),
                abs(lhs.commandedTipTarget.y - rhs.commandedTipTarget.y)].max()!
    }
}

private struct PhysicalFlowRandomNumberGenerator {
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
        return Float(Double(z >> 11)
            * (1.0 / 9_007_199_254_740_992.0))
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
