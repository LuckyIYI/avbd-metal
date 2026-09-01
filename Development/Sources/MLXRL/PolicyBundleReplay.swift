import Foundation
import SimCore
import PhysicsAVBD
import RL

public enum PolicyBundleTrust: Sendable, Equatable {
    case qualified
    case externalParityVerified
    case importedUnverified

    public var displayLabel: String {
        switch self {
        case .qualified: "QUALIFIED PACKAGED POLICY"
        case .externalParityVerified: "VERIFIED PACKAGED EXTERNAL POLICY"
        case .importedUnverified: "UNVERIFIED POLICY BUNDLE"
        }
    }

    public var isVerified: Bool { self != .importedUnverified }
}

/// Runtime-neutral surface consumed by Policy Replay. Implementations are
/// selected by the bundle's versioned runtime ABI, never by bundle id, task
/// name, or UI policy selection.
public protocol PolicyBundleReplaySession: AnyObject {
    var solver: GPUSolver { get }
    var capabilities: RLReplayCapabilities { get }
    var controlStepSeconds: Float { get }
    var episodeFinished: Bool { get }
    var controlSteps: Int { get }
    var values: [String: Float] { get }

    func reset() throws
    func step() throws
    func anchor(named name: String) -> F3?
    func perform(command: String, arguments: [String: Float]) throws
}

public enum PolicyBundleReplayFactory {
    public static func make(
        bundle: LoadedPolicyBundle,
        release: PolicyBundleReleaseIndex.Release? = nil
    ) throws -> any PolicyBundleReplaySession {
        let session: any PolicyBundleReplaySession = switch bundle.manifest.runtime.kind {
        case .vectorPPO:
            try VectorPPOBundleReplaySession(
                bundle: bundle, release: release)
        case .unitreeH1Recurrent:
            try UnitreeH1BundleReplaySession(
                bundle: bundle, release: release)
        }
        try validatePresentation(
            bundle.manifest.presentation, capabilities: session.capabilities)
        return session
    }

    static func validatePresentation(
        _ presentation: PolicyBundleManifest.Presentation,
        capabilities: RLReplayCapabilities
    ) throws {
        for camera in presentation.cameraPresets
            where !capabilities.anchors.contains(camera.anchor) {
            throw PolicyBundleError.invalid(
                "camera '\(camera.id)' uses unsupported anchor "
                    + "'\(camera.anchor)'")
        }
        for metric in presentation.metrics
            where !capabilities.values.contains(metric.source)
                && !isDynamicStepMetricSource(metric.source) {
            throw PolicyBundleError.invalid(
                "metric '\(metric.id)' uses unsupported source "
                    + "'\(metric.source)'")
        }
        for control in presentation.controls {
            if let command = control.command,
               !capabilities.commands.contains(command) {
                throw PolicyBundleError.invalid(
                    "control '\(control.id)' uses unsupported command "
                        + "'\(command)'")
            }
        }
    }

    /// Step-metric names are produced dynamically by `RLStepBatch`, so a
    /// task cannot enumerate them before its first transition. Keep their
    /// namespace data-driven while rejecting empty or malformed paths.
    static func isDynamicStepMetricSource(_ source: String) -> Bool {
        let prefix = "metric/"
        guard source.hasPrefix(prefix) else { return false }
        let suffix = source.dropFirst(prefix.count)
        let components = suffix.split(
            separator: "/", omittingEmptySubsequences: false)
        guard !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty }) else { return false }
        return suffix.utf8.allSatisfy { byte in
            (byte >= 48 && byte <= 57)
                || (byte >= 65 && byte <= 90)
                || (byte >= 97 && byte <= 122)
                || byte == 45 || byte == 46 || byte == 47 || byte == 95
        }
    }

    /// Resolve the declarative camera target. `target` is an absolute point
    /// for the world anchor and an anchor-relative vector otherwise; `offset`
    /// is applied in both cases.
    package static func cameraTarget(
        preset: PolicyBundleManifest.CameraPreset,
        anchorValue: F3?
    ) -> F3? {
        let target = F3(
            preset.target[0], preset.target[1], preset.target[2])
        let offset = F3(
            preset.offset[0], preset.offset[1], preset.offset[2])
        if preset.anchor == "world" { return target + offset }
        guard let anchorValue else { return nil }
        return anchorValue + target + offset
    }
}

private final class VectorPPOBundleReplaySession:
    PolicyBundleReplaySession
{
    let bundle: LoadedPolicyBundle
    let task: any VectorizedRLTask
    let replayTask: any RLReplayTask
    let provider: any RLActionProvider

    private var observation: RLObservationBatch
    private var result: RLStepBatch
    private var seed: UInt64

    private(set) var episodeFinished = false
    private(set) var controlSteps = 0
    private(set) var values = [String: Float]()

    var solver: GPUSolver { replayTask.replaySolver }
    var capabilities: RLReplayCapabilities {
        var capabilities = replayTask.replayCapabilities
        capabilities.values.formUnion([
            "replay/control-step", "replay/reward", "replay/success",
            "replay/finished",
        ])
        return capabilities
    }
    var controlStepSeconds: Float { task.spec.controlStep }

    init(
        bundle: LoadedPolicyBundle,
        release: PolicyBundleReleaseIndex.Release?
    ) throws {
        self.bundle = bundle
        let simulation = bundle.manifest.simulation
        let options = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: simulation.task,
            semanticOptions: simulation.options,
            maxEpisodeSteps: simulation.maxEpisodeSteps,
            controlDecimation: simulation.controlDecimation)
        let configured = try BuiltInRLTasks.registry.make(
            simulation.task,
            configuration: RLTaskConfiguration(
                numEnvironments: 1,
                seed: simulation.seed,
                autoReset: false,
                includeInteractiveRobustnessProbes:
                    simulation.includeInteractiveRobustnessProbes,
                options: options))
        guard let presentable = configured as? any RLReplayTask else {
            throw PolicyBundleError.invalid(
                "task '\(simulation.task)' does not implement the replay ABI")
        }
        guard configured.spec.revision == simulation.taskRevision,
              configured.spec.maxEpisodeSteps == simulation.maxEpisodeSteps,
              configured.spec.simulationStep
                == simulation.simulationStepSeconds,
              configured.spec.controlDecimation
                == simulation.controlDecimation,
              configured.spec.configurationValues == simulation.options else {
            throw PolicyBundleError.invalid(
                "constructed task disagrees with bundle simulation contract")
        }

        // An imported actor is not qualified, but it must still be internally
        // self-consistent. This runtime verifies the candidate deployment
        // manifest, all checkpoint digests, metadata, and training state;
        // optional release anchors add independent commissioning.
        let loadedProvider: any RLActionProvider =
            try VectorPolicyDeploymentRuntime(
                bundleDirectory: bundle.directory.path,
                expectedTask: simulation.task,
                expectedTaskRevision:
                    release?.expectedTaskRevision ?? simulation.taskRevision,
                expectedCheckpointFingerprint:
                    release?.expectedCheckpointFingerprint,
                expectedDeploymentManifestSHA256:
                    release?.expectedDeploymentManifestSHA256)
        task = configured
        replayTask = presentable
        provider = loadedProvider
        observation = RLObservationBatch(spec: configured.spec)
        result = RLStepBatch(spec: configured.spec)
        seed = simulation.seed
        try reset()
    }

    func reset() throws {
        observation = try task.reset(seed: seed)
        try provider.reset(for: task, observation: observation)
        result = RLStepBatch(spec: task.spec)
        episodeFinished = false
        controlSteps = 0
        refreshValues()
    }

    func step() throws {
        guard !episodeFinished else { return }
        let actions = try provider.actions(for: observation, task: task)
        result = try task.step(actions: actions)
        observation = result.observations
        try provider.resetAfterStep(for: task, result: result)
        controlSteps += 1
        episodeFinished = result.terminated[0] || result.truncated[0]
        refreshValues()
    }

    func anchor(named name: String) -> F3? {
        replayTask.replayAnchor(named: name, environment: 0)
    }

    func perform(command: String, arguments: [String: Float]) throws {
        let effect = try replayTask.performReplayCommand(
            command, arguments: arguments, environment: 0)
        if effect == .reset {
            seed &+= 0x9E3779B97F4A7C15
            try reset()
        } else {
            refreshValues()
        }
    }

    private func refreshValues() {
        values = replayTask.replayValues(
            environment: 0, latestStep: result)
        values["replay/control-step"] = Float(controlSteps)
        values["replay/reward"] = result.rewards.first ?? 0
        values["replay/success"] = result.successes.first == true ? 1 : 0
        values["replay/finished"] = episodeFinished ? 1 : 0
    }
}

private final class UnitreeH1BundleReplaySession:
    PolicyBundleReplaySession
{
    let bundle: LoadedPolicyBundle
    let release: PolicyBundleReleaseIndex.Release?
    private var session: UnitreeH1Sim2SimSession
    private let command: SIMD3<Float>
    private let maxEpisodeSteps: Int
    private let solverIterations: Int

    private(set) var episodeFinished = false
    private(set) var values = [String: Float]()

    var solver: GPUSolver { session.environment.solver }
    var capabilities: RLReplayCapabilities {
        .init(
            anchors: ["robot", "course", "world"],
            values: [
                "task/root-x", "task/root-y", "task/root-height",
                "task/planar-speed", "task/yaw-rate", "task/upright",
                "task/command-forward", "task/command-lateral",
                "task/command-yaw", "replay/control-step",
                "replay/finished",
            ],
            commands: ["throw-projectile"])
    }
    var controlStepSeconds: Float {
        session.policy.manifest.control.physicsTimeStep
            * Float(session.policy.manifest.control.controlDecimation)
    }
    var controlSteps: Int { session.controlSteps }

    init(
        bundle: LoadedPolicyBundle,
        release: PolicyBundleReleaseIndex.Release?
    ) throws {
        self.bundle = bundle
        self.release = release
        let options = bundle.manifest.simulation.options
        command = SIMD3(
            options["commandX"] ?? 0.5,
            options["commandY"] ?? 0,
            options["commandYaw"] ?? 0)
        maxEpisodeSteps = bundle.manifest.simulation.maxEpisodeSteps
        guard let configuredIterations = options["solverIterations"]
                .flatMap(Int.init(exactly:)),
              configuredIterations > 0 else {
            throw PolicyBundleError.invalid(
                "Unitree replay requires positive integer solverIterations")
        }
        solverIterations = configuredIterations
        session = try Self.makeSession(
            bundle: bundle, release: release, command: command,
            solverIterations: solverIterations)
        refreshValues()
    }

    func reset() throws {
        session = try Self.makeSession(
            bundle: bundle, release: release, command: command,
            solverIterations: solverIterations)
        episodeFinished = false
        refreshValues()
    }

    func step() throws {
        guard !episodeFinished else { return }
        let state = try session.step()
        let upright = state.root.rotation.act(F3(0, 0, 1)).z
        episodeFinished = controlSteps >= maxEpisodeSteps
            || state.root.position.z < 0.55 || upright < 0.5
        refreshValues(state: state)
    }

    func anchor(named name: String) -> F3? {
        switch name {
        case "robot": return session.environment.state().root.position
        case "course":
            return session.environment.state().root.position + F3(6, 0, 0)
        case "world": return .zero
        default: return nil
        }
    }

    func perform(command name: String, arguments: [String: Float]) throws {
        switch name {
        case "throw-projectile":
            session.environment.throwBoxes(
                environmentIDs: [0],
                sideSigns: [arguments["side"] ?? 1],
                speed: arguments["speed"] ?? 6)
            refreshValues()
        default:
            throw PolicyBundleError.invalid(
                "runtime does not support command '\(name)'")
        }
    }

    private func refreshValues(state: HumanoidState? = nil) {
        let state = state ?? session.environment.state()
        let upright = state.root.rotation.act(F3(0, 0, 1)).z
        values = [
            "task/root-x": state.root.position.x,
            "task/root-y": state.root.position.y,
            "task/root-height": state.root.position.z,
            "task/planar-speed": hypot(
                state.root.linearVelocity.x, state.root.linearVelocity.y),
            "task/yaw-rate": state.root.angularVelocity.z,
            "task/upright": upright,
            "task/command-forward": command.x,
            "task/command-lateral": command.y,
            "task/command-yaw": command.z,
            "replay/control-step": Float(controlSteps),
            "replay/finished": episodeFinished ? 1 : 0,
        ]
    }

    private static func makeSession(
        bundle: LoadedPolicyBundle,
        release: PolicyBundleReleaseIndex.Release?,
        command: SIMD3<Float>,
        solverIterations: Int
    ) throws -> UnitreeH1Sim2SimSession {
        try UnitreeH1Sim2SimSession(
            policyDirectory: bundle.directory.path,
            command: command,
            solverIterations: solverIterations,
            expectedReleaseIdentity: release?.unitreeH1ReleaseIdentity)
    }
}
