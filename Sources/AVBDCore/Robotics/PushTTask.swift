import simd

public struct PushTTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
    public var actionScale: Float
    public var progressWeight: Float
    public var reachProgressWeight: Float
    public var actionRateWeight: Float
    public var successBonus: Float
    public var positionTolerance: Float
    public var yawTolerance: Float
    public var autoReset: Bool

    public init(numEnvironments: Int, seed: UInt64 = 1,
                maxEpisodeSteps: Int = 300, controlDecimation: Int = 4,
                actionScale: Float = 0.18, progressWeight: Float = 8,
                reachProgressWeight: Float = 0.5,
                actionRateWeight: Float = -0.01, successBonus: Float = 10,
                positionTolerance: Float = 0.25, yawTolerance: Float = 0.35,
                autoReset: Bool = true) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
        self.actionScale = actionScale
        self.progressWeight = progressWeight
        self.reachProgressWeight = reachProgressWeight
        self.actionRateWeight = actionRateWeight
        self.successBonus = successBonus
        self.positionTolerance = positionTolerance
        self.yawTolerance = yawTolerance
        self.autoReset = autoReset
    }
}

/// State-based Push-T benchmark on the common vector task contract. The
/// environment and reward code know about Push-T; the solver and PPO runner
/// do not. This is the reference pattern for adding manipulation tasks.
public final class PushTTask: VectorizedRLTask {
    public let spec: RLTaskSpec
    public let environment: PushTEnv
    public let configuration: PushTTaskConfig

    private var targets: [SIMD2<Float>]
    private var previousActions: ContiguousArray<Float>
    private var previousGoalDistance: [Float]
    private var previousReachDistance: [Float]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var resetRNG: SplitMix64

    public init(configuration: PushTTaskConfig) throws {
        guard configuration.numEnvironments > 0 else {
            throw RLEnvironmentError.invalidConfiguration("numEnvironments must be positive")
        }
        guard configuration.maxEpisodeSteps > 0, configuration.controlDecimation > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "episode length and control decimation must be positive")
        }
        guard configuration.actionScale > 0 else {
            throw RLEnvironmentError.invalidConfiguration("actionScale must be positive")
        }
        let env = try PushTEnv(numEnvs: configuration.numEnvironments,
                               seed: configuration.seed)
        environment = env
        self.configuration = configuration
        spec = RLTaskSpec(
            id: "pusht-state-v0",
            revision: RLPhysicsContract.deterministicColorSolveV1(1),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(name: "policy", shape: [12]),
            action: RLTensorSpec(name: "tip_delta", shape: [2],
                                 lowerBound: [-1, -1], upperBound: [1, 1]),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: configuration.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: [
                "actionScale": configuration.actionScale,
                "progressWeight": configuration.progressWeight,
                "reachProgressWeight": configuration.reachProgressWeight,
                "actionRateWeight": configuration.actionRateWeight,
                "successBonus": configuration.successBonus,
                "positionTolerance": configuration.positionTolerance,
                "yawTolerance": configuration.yawTolerance,
            ])
        targets = [SIMD2<Float>](repeating: .zero,
                                count: configuration.numEnvironments)
        previousActions = ContiguousArray(repeating: 0,
                                          count: configuration.numEnvironments * 2)
        previousGoalDistance = [Float](repeating: 0,
                                       count: configuration.numEnvironments)
        previousReachDistance = [Float](repeating: 0,
                                        count: configuration.numEnvironments)
        episodeLengths = [Int](repeating: 0, count: configuration.numEnvironments)
        episodeReturns = [Float](repeating: 0, count: configuration.numEnvironments)
        resetRNG = SplitMix64(seed: configuration.seed &+ 0xA0761D6478BD642F)

        let states = env.states()
        initializeEpisodeState(Array(0..<configuration.numEnvironments), states: states)
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        try environment.solver.synchronize()
        let seeds = envIDs.map { seed &+ UInt64($0) &* 0x9E3779B97F4A7C15 }
        environment.reset(envIDs, seeds: seeds)
        let states = environment.states()
        initializeEpisodeState(envIDs, states: states)
        fillObservations(states, into: &observations.policy)
        try observations.validate(for: spec)
    }

    public func step(actions: RLActionBatch, into result: inout RLStepBatch) throws {
        try result.validate(for: spec)
        try actions.validate(for: spec)
        try environment.solver.synchronize()
        result.clearSignals()
        let n = spec.numEnvironments
        var actionRate = ContiguousArray(repeating: Float(0), count: n)
        for e in 0..<n {
            let ax = simd_clamp(actions[e, 0], -1, 1)
            let ay = simd_clamp(actions[e, 1], -1, 1)
            let oldX = previousActions[e * 2]
            let oldY = previousActions[e * 2 + 1]
            actionRate[e] = (ax - oldX) * (ax - oldX) + (ay - oldY) * (ay - oldY)
            previousActions[e * 2] = ax
            previousActions[e * 2 + 1] = ay
            targets[e] += SIMD2(ax, ay) * configuration.actionScale
            targets[e] = simd_clamp(targets[e], SIMD2(repeating: -3),
                                    SIMD2(repeating: 3))
        }

        try environment.stepChecked(
            actions: targets, substeps: configuration.controlDecimation,
            maxStep: configuration.actionScale)
        var states = environment.states()
        fillObservations(states, into: &result.observations.policy)

        var goalProgress = ContiguousArray(repeating: Float(0), count: n)
        var reachProgress = ContiguousArray(repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var resetIDs = [Int]()
        var resetSeeds = [UInt64]()

        for e in 0..<n {
            let goal = environment.refs[e].goalPos
            let goalDistance = length(goal - states[e].blockPosition)
            let reachDistance = length(states[e].tipPosition - states[e].blockPosition)
            goalProgress[e] = previousGoalDistance[e] - goalDistance
            reachProgress[e] = previousReachDistance[e] - reachDistance
            let success = environment.success(e,
                                              posTol: configuration.positionTolerance,
                                              yawTol: configuration.yawTolerance)
            episodeLengths[e] += 1
            let timedOut = episodeLengths[e] >= configuration.maxEpisodeSteps
            var reward = configuration.progressWeight * goalProgress[e]
                + configuration.reachProgressWeight * reachProgress[e]
                + configuration.actionRateWeight * actionRate[e]
                - 0.002 * goalDistance
            if success { reward += configuration.successBonus }
            result.rewards[e] = reward
            result.successes[e] = success
            result.terminated[e] = success
            result.truncated[e] = !success && timedOut
            episodeReturns[e] += reward
            previousGoalDistance[e] = goalDistance
            previousReachDistance[e] = reachDistance

            if success || timedOut {
                result.hasFinalObservation[e] = true
                let row = e * spec.observation.elementCount
                for j in 0..<spec.observation.elementCount {
                    result.finalObservations[row + j] = result.observations.policy[row + j]
                }
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            }
        }

        result.metrics["reward/goal_progress"] = goalProgress
        result.metrics["reward/reach_progress"] = reachProgress
        result.metrics["penalty/action_rate"] = actionRate
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric

        if !resetIDs.isEmpty {
            environment.reset(resetIDs, seeds: resetSeeds)
            states = environment.states()
            initializeEpisodeState(resetIDs, states: states)
            fillObservations(states, into: &result.observations.policy)
        }
        try result.observations.validate(for: spec)
    }

    private func initializeEpisodeState(_ ids: [Int], states: [PushTState]) {
        for e in ids {
            targets[e] = states[e].tipPosition
            previousActions[e * 2] = 0
            previousActions[e * 2 + 1] = 0
            previousGoalDistance[e] = length(environment.refs[e].goalPos
                                              - states[e].blockPosition)
            previousReachDistance[e] = length(states[e].tipPosition
                                               - states[e].blockPosition)
            episodeLengths[e] = 0
            episodeReturns[e] = 0
        }
    }

    private func fillObservations(_ states: [PushTState],
                                  into output: inout ContiguousArray<Float>) {
        let d = spec.observation.elementCount
        for e in 0..<spec.numEnvironments {
            let s = states[e]
            let goalDelta = environment.refs[e].goalPos - s.blockPosition
            let o = e * d
            output[o] = s.tipPosition.x / 3.25
            output[o + 1] = s.tipPosition.y / 3.25
            output[o + 2] = s.tipVelocity.x / 5
            output[o + 3] = s.tipVelocity.y / 5
            output[o + 4] = s.blockPosition.x / 3.25
            output[o + 5] = s.blockPosition.y / 3.25
            output[o + 6] = sin(s.blockYaw)
            output[o + 7] = cos(s.blockYaw)
            output[o + 8] = goalDelta.x / 6.5
            output[o + 9] = goalDelta.y / 6.5
            output[o + 10] = previousActions[e * 2]
            output[o + 11] = previousActions[e * 2 + 1]
        }
    }
}
