import SwiftUI
import Combine
import MetalKit
import AVBDCore
import AVBDLearn

/// Visual, real-time replay for the exact VectorizedRLTask/checkpoint pair used
/// by headless evaluation. There is no separate hand-authored display scene.
@MainActor
final class PolicyReplayModel: ObservableObject, RenderableModel {
    nonisolated let captureID = "policy"
    enum Robot: String, CaseIterable {
        case unitreeH1 = "Unitree H1 Sim2Sim"
        case humanoidIsaac = "H1 Flat Walk"
        case humanoidIsaacGoal = "H1 Goal"
        case humanoidWalk = "Humanoid Walk"
        case humanoidGoal = "Legacy Goal"
        case arm = "Arm Push-T"
        case arachne = "Arachne-15"
        case arachneGoal = "Arachne Goal"
        case arachneClassical = "Arachne Classical"

        var taskID: String {
            switch self {
            case .unitreeH1: return "unitree-h1-sim2sim-v0"
            case .humanoidIsaac: return "humanoid-isaac-flat-v0"
            case .humanoidIsaacGoal: return "humanoid-isaac-goal-v0"
            case .humanoidWalk: return "humanoid-walk-v0"
            case .humanoidGoal: return "humanoid-goal-v0"
            case .arm: return "arm-pusht-v0"
            case .arachne: return "arachne15-velocity-v0"
            case .arachneGoal, .arachneClassical: return "arachne15-goal-v0"
            }
        }

        var selectionID: String {
            self == .arachneClassical
                ? "arachne15-classical-goal-v0" : taskID
        }

        var usesClassicalController: Bool { self == .arachneClassical }

        static func fromSelectionID(_ id: String) -> Robot? {
            if id == "arachne15-classical-goal-v0" {
                return .arachneClassical
            }
            return Robot(taskID: id)
        }

        init?(taskID: String) {
            switch taskID {
            case "unitree-h1-sim2sim-v0": self = .unitreeH1
            case "humanoid-isaac-flat-v0": self = .humanoidIsaac
            case "humanoid-isaac-goal-v0": self = .humanoidIsaacGoal
            case "humanoid-walk-v0": self = .humanoidWalk
            case "humanoid-goal-v0": self = .humanoidGoal
            case "arm-pusht-v0": self = .arm
            case "arachne15-velocity-v0": self = .arachne
            case "arachne15-goal-v0": self = .arachneGoal
            default: return nil
            }
        }
    }
    enum CameraMode: String, CaseIterable {
        case follow = "Follow Robot"
        case course = "Show Course"
    }

    @Published var robot: Robot = {
        let environment = ProcessInfo.processInfo.environment
        if environment["AVBD_REPLAY_CONTROLLER"]?.lowercased()
            == "classical" { return .arachneClassical }
        let requestedTask = environment["AVBD_REPLAY_TASK"]
            ?? UserDefaults.standard.string(
                forKey: "AVBDPolicyReplaySelectedTask")
        return requestedTask.flatMap(Robot.fromSelectionID)
            ?? (environment["AVBD_REPLAY_ROBOT"] == "arm"
                ? .arm : .humanoidIsaac)
    }() {
        didSet {
            UserDefaults.standard.set(
                robot.selectionID, forKey: "AVBDPolicyReplaySelectedTask")
            rebuild()
        }
    }
    @Published var cameraMode: CameraMode = .follow {
        didSet {
            applyCameraPreset()
            cameraEpoch += 1
        }
    }
    @Published var running = true
    @Published var playbackRate: Double = 1
    @Published var statsText = ""
    @Published var policyStatus = ""
    @Published var trainingStatus = "waiting for trainer metrics"
    @Published var interactionStatus = ""
    @Published var goalBearingDegrees: Double = 0
    @Published var goalDistance: Double = 8
    @Published var boxSpeed: Double = 6
    @Published var autoLoadLatest: Bool = {
        guard let value = ProcessInfo.processInfo.environment[
            "AVBD_REPLAY_AUTO_LOAD_LATEST"]?.lowercased() else { return true }
        return !["0", "false", "no", "off"].contains(value)
    }()
    @Published private(set) var loadedUpdate: Int?
    @Published private(set) var newestUpdate: Int?
    @Published private(set) var episodeFinished = false
    @Published private(set) var replayCameraTarget = F3(0, 0, 1)
    @Published private(set) var courseCameraTarget = F3(6.5, 0, 0.9)
    var colorByGraphColor = false
    private(set) var cameraEpoch = 0

    private var isaacHumanoid: HumanoidIsaacVelocityTask?
    private var humanoid: HumanoidWalkTask?
    private var arm: ArmPushTTask?
    private var arachne: Arachne15LocomotionTask?
    private var unitreeH1: UnitreeH1Sim2SimSession?
    private var task: (any VectorizedRLTask)?
    private var observation: RLObservationBatch?
    private var result: RLStepBatch?
    private var actionProvider: (any RLActionProvider)?
    private var loadedCheckpointDirectory: String?
    private let liveRunDirectory = ProcessInfo.processInfo.environment[
        "AVBD_REPLAY_RUN_DIR"]
    private var accumulator: Double = 0
    private var lastTime = CACurrentMediaTime()
    /// Offline evidence capture must map one rendered image to one policy
    /// transition. PNG readback is much slower than real-time rendering; the
    /// interactive wall-clock catch-up path would otherwise skip simulator
    /// states and pad the end of the video with a frozen terminal pose.
    private let frameLockedCapture = ProcessInfo.processInfo.environment[
        "AVBD_REPLAY_FRAME_LOCK"] != nil
    private var controlSteps = 0
    private var completed = 0
    private var successes = 0
    private var nextProjectileSide: Float = 1
    private var replaySeed: UInt64 = 21_001
    private var unitreeInitialPosition = F3.zero

    var supportsGoalPlacement: Bool {
        isaacHumanoid?.usesPointGoal == true
            || humanoid?.usesPointGoal == true
            || arachne?.usesPointGoal == true
    }
    var usesCheckpoint: Bool { !robot.usesClassicalController }
    var controllerSectionTitle: String {
        usesCheckpoint ? "Parallel training" : "Controller"
    }
    var goalControlExplanation: String {
        usesCheckpoint
            ? "Changing the goal updates the learned policy's command; it does not steer the joints directly."
            : "Changing the goal updates body-twist steering. The CPG and IK produce joint targets; only motor torque and physical contacts move the body."
    }
    var goalDistanceRange: ClosedRange<Double> {
        arachne?.usesPointGoal == true ? 0.4...3.0 : 2...12
    }
    var supportsBoxThrows: Bool {
        isaacHumanoid?.hasProjectile(environment: 0) == true
            || humanoid?.hasProjectile(environment: 0) == true
    }
    var impactModelSummary: String {
        if let isaacHumanoid, isaacHumanoid.hasProjectile(environment: 0) {
            return String(
                format: "%.1f kg box · full imported H1 collision primitives",
                isaacHumanoid.environment.projectileMass)
        }
        if let humanoid, humanoid.hasProjectile(environment: 0) {
            return String(
                format: "%.1f kg box · articulated robot collision bodies",
                humanoid.environment.projectileMass)
        }
        return "no physical projectile in this checkpoint scene"
    }
    var scenarioSummary: String {
        switch robot {
        case .unitreeH1:
            return "Unchanged public Unitree RL Gym H1 recurrent policy, imported from TorchScript to MLX and running on AVBD Metal physics."
        case .humanoidIsaac:
            return "Replay the accepted H1 policy on the exact public Flat velocity task. Markers show the current command segment, not a point goal."
        case .humanoidIsaacGoal:
            return "Imported H1 point-goal task transferred from the accepted Flat locomotion policy; PPO owns every joint action."
        case .humanoidWalk:
            return "One complete A→B evaluation of a learned velocity policy."
        case .humanoidGoal:
            return "Experimental native-humanoid goal policy; this is not the accepted imported-H1 Flat policy."
        case .arm:
            return "Replay the learned articulated-arm Push-T policy."
        case .arachne:
            return "Arachne-15 velocity locomotion with the exact printable CAD visuals, explicit training colliders, measured mass budget, actuator limits, latency, and seeded plant variation."
        case .arachneGoal:
            return "Arachne-15 samples a random world target, converts it to the reusable local velocity/yaw command, and must enter the visible target slowly enough to stop there."
        case .arachneClassical:
            return "Non-neural paired-ripple CPG and exact two-axis leg IK drive the same torque-limited motors, contacts, randomized plant, and arbitrary point-goal task as the learned policy."
        }
    }

    var solver: GPUSolver? {
        if let unitreeH1 { return unitreeH1.environment.solver }
        if let isaacHumanoid { return isaacHumanoid.environment.solver }
        if let humanoid { return humanoid.environment.solver }
        if let arm { return arm.environment.solver }
        return arachne?.environment.solver
    }

    init() {
        // An explicit checkpoint is authoritative. This prevents persisted UI
        // selection/restoration from constructing a different task and then
        // rejecting the requested policy as incompatible.
        if let path = ProcessInfo.processInfo.environment[
                "AVBD_REPLAY_CHECKPOINT"],
           let data = try? Data(contentsOf: URL(fileURLWithPath: path)
                .appendingPathComponent("metadata.json")),
           let metadata = try? JSONDecoder().decode(
                VectorPolicyMetadata.self, from: data),
           let checkpointRobot = Robot(taskID: metadata.task) {
            robot = checkpointRobot
        }
        rebuild()
    }

    func rebuild() {
        unitreeH1 = nil; isaacHumanoid = nil; humanoid = nil; arm = nil
        arachne = nil
        task = nil; actionProvider = nil
        loadedCheckpointDirectory = nil; loadedUpdate = nil; newestUpdate = nil
        completed = 0; successes = 0; controlSteps = 0; accumulator = 0
        replaySeed = 21_001
        interactionStatus = ""
        episodeFinished = false
        running = true
        do {
            if robot == .unitreeH1 {
                try installUnitreeH1Replay()
                updateCameraTargets()
                applyCameraPreset()
                cameraEpoch += 1
                lastTime = CACurrentMediaTime()
                refreshStats()
                return
            }
            let desiredTaskID = robot.taskID
            let packagedPath = "checkpoints/\(desiredTaskID)"
            let bundledPath = Bundle.main.resourceURL?
                .appendingPathComponent("checkpoints/\(desiredTaskID)").path
            let overridePath = ProcessInfo.processInfo.environment[
                "AVBD_REPLAY_CHECKPOINT"]
            let configurationPaths = robot.usesClassicalController ? [] : [
                autoLoadLatest ? liveRunDirectory : nil,
                overridePath, bundledPath, packagedPath,
            ].compactMap { $0 }
            let checkpointMetadata = configurationPaths.lazy.compactMap {
                self.replayMetadata(at: $0, task: desiredTaskID)
            }.first
            var replayOptions = checkpointMetadata?.taskConfiguration ?? [:]
            if let checkpointMetadata {
                // These shape the task but live in the checkpoint's structural
                // metadata rather than taskConfiguration. Restore them before
                // constructing replay so non-default curricula load exactly.
                replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
                    for: desiredTaskID,
                    semanticOptions: replayOptions,
                    maxEpisodeSteps: checkpointMetadata.maxEpisodeSteps,
                    controlDecimation: checkpointMetadata.controlDecimation)
            }
            let configuredTask = try BuiltInRLTasks.registry.make(
                desiredTaskID,
                configuration: RLTaskConfiguration(
                    numEnvironments: 1, seed: 21_001, autoReset: false,
                    options: replayOptions))
            isaacHumanoid = configuredTask as? HumanoidIsaacVelocityTask
            humanoid = configuredTask as? HumanoidWalkTask
            arm = configuredTask as? ArmPushTTask
            arachne = configuredTask as? Arachne15LocomotionTask
            task = configuredTask
            if robot.usesClassicalController {
                guard arachne?.usesPointGoal == true else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "classical Arachne controller requires point-goal task")
                }
                actionProvider = Arachne15ClassicalController()
                policyStatus = "non-neural controller · paired-ripple CPG · exact IK · no checkpoint"
            }
            if arachne?.usesPointGoal == true,
               !goalDistanceRange.contains(goalDistance) {
                goalDistance = 1.5
            }
            if let humanoid, humanoid.usesPointGoal {
                try installSelectedGoal(in: humanoid)
            }
            guard let task else { return }
            let liveCheckpoint = robot.usesClassicalController ? nil
                : liveRunDirectory.flatMap {
                VectorPolicyCheckpointDiscovery.latestCompleteCheckpoint(
                    inRunDirectory: $0, task: task.spec.id,
                    taskRevision: task.spec.revision)
            }
            newestUpdate = liveCheckpoint?.completedUpdates
            let startupLiveCheckpoint = autoLoadLatest ? liveCheckpoint : nil
            let candidates = robot.usesClassicalController ? [] : [
                startupLiveCheckpoint?.directory, overridePath,
                bundledPath, packagedPath,
            ].compactMap { $0 }
            var loadFailures = [String]()
            for path in candidates
                where FileManager.default.fileExists(atPath: "\(path)/metadata.json") {
                do {
                    try installCheckpoint(
                        path, candidate: path == startupLiveCheckpoint?.directory
                            ? startupLiveCheckpoint : nil)
                    break
                } catch {
                    loadFailures.append("\(path): \(error)")
                }
            }
            if actionProvider == nil {
                running = false
                policyStatus = loadFailures.isEmpty
                    ? "no promoted checkpoint for \(task.spec.id)"
                    : "no compatible checkpoint: \(loadFailures.joined(separator: "; "))"
            }
            try restartEpisode()
            refreshTrainingMetrics()
        } catch {
            policyStatus = "replay unavailable: \(error.localizedDescription)"
        }
        updateCameraTargets()
        applyCameraPreset()
        cameraEpoch += 1
        lastTime = CACurrentMediaTime()
        refreshStats()
    }

    private func installUnitreeH1Replay() throws {
        let overridePath = ProcessInfo.processInfo.environment[
            "AVBD_REPLAY_CHECKPOINT"]
        let bundledPath = Bundle.main.resourceURL?
            .appendingPathComponent("checkpoints/external/unitree-h1").path
        let candidates = [overridePath, bundledPath,
                          "checkpoints/external/unitree-h1"].compactMap { $0 }
        guard let directory = candidates.first(where: {
            FileManager.default.fileExists(atPath: "\($0)/manifest.json")
                && FileManager.default.fileExists(
                    atPath: "\($0)/policy.safetensors")
        }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "imported Unitree H1 policy not found; run "
                    + "Tools/import_unitree_h1_policy.py first")
        }
        let session = try UnitreeH1Sim2SimSession(policyDirectory: directory)
        unitreeH1 = session
        loadedCheckpointDirectory = directory
        unitreeInitialPosition = session.environment.state().root.position
        controlSteps = 0
        episodeFinished = false
        running = true
        policyStatus = "loaded verified Unitree checkpoint "
            + String(session.policy.manifest.source.checkpointSHA256.prefix(12))
            + "…\n\(directory)"
        trainingStatus = "external pretrained policy · recurrent MLX inference"
    }

    /// Configure the simulator from the checkpoint's serialized task options
    /// before compatibility checks. This keeps live replay on the exact task
    /// used by training, including gated actors and future task-owned options.
    private func replayMetadata(
        at directory: String, task desiredTask: String
    ) -> VectorPolicyMetadata? {
        guard let metadata = try? JSONDecoder().decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath: directory)
                .appendingPathComponent("metadata.json"))),
              metadata.task == desiredTask else { return nil }
        return metadata
    }

    func togglePlayback() {
        if episodeFinished {
            resetEpisode()
        } else {
            running.toggle()
            lastTime = CACurrentMediaTime()
        }
    }

    func resetEpisode() {
        do {
            try restartEpisode()
        } catch {
            policyStatus = "reset failed: \(error.localizedDescription)"
            running = false
        }
    }

    func applyGoalAndReset() {
        do {
            if let isaacHumanoid, isaacHumanoid.usesPointGoal {
                try installSelectedGoal(in: isaacHumanoid)
            } else if let humanoid, humanoid.usesPointGoal {
                try installSelectedGoal(in: humanoid)
            } else if let arachne, arachne.usesPointGoal {
                try installSelectedGoal(in: arachne)
            } else {
                return
            }
            try restartEpisode()
            interactionStatus = String(
                format: "goal set: %.1f m at %+.0f°",
                goalDistance, goalBearingDegrees)
        } catch {
            policyStatus = "goal update failed: \(error.localizedDescription)"
            running = false
        }
    }

    func randomizeGoalAndReset() {
        do {
            if let isaacHumanoid, isaacHumanoid.usesPointGoal {
                isaacHumanoid.clearGoalOverride(environment: 0)
            } else if let humanoid, humanoid.usesPointGoal {
                humanoid.clearGoalOverride(environment: 0)
            } else if let arachne, arachne.usesPointGoal {
                arachne.clearGoalOverride(environment: 0)
            } else {
                return
            }
            replaySeed &+= 0x9E3779B97F4A7C15
            try restartEpisode()
            let direction: F3
            let distance: Float
            if isaacHumanoid?.usesPointGoal == true {
                direction = isaacHumanoid!.currentGoalDirection(environment: 0)
                distance = simd_length(
                    isaacHumanoid!.currentGoalPosition(environment: 0)
                        - isaacHumanoid!.environment.states()[0].root.position)
            } else if humanoid?.usesPointGoal == true {
                direction = humanoid!.currentGoalDirection(environment: 0)
                distance = simd_length(
                    humanoid!.currentGoalPosition(environment: 0)
                        - humanoid!.environment.states()[0].root.position)
            } else {
                direction = arachne!.currentGoalDirection(environment: 0)
                distance = arachne!.currentGoalDistance(environment: 0)
            }
            goalBearingDegrees = Double(atan2(direction.y, direction.x) * 180 / .pi)
            goalDistance = Double(distance)
            interactionStatus = "goal sampled from the checkpoint task distribution"
        } catch {
            policyStatus = "goal reset failed: \(error.localizedDescription)"
            running = false
        }
    }

    private func installSelectedGoal(in humanoid: HumanoidWalkTask) throws {
        let bearing = Float(goalBearingDegrees * .pi / 180)
        try humanoid.setGoalOverride(
            environment: 0,
            direction: F3(cos(bearing), sin(bearing), 0),
            distance: Float(goalDistance))
    }

    private func installSelectedGoal(
        in humanoid: HumanoidIsaacVelocityTask
    ) throws {
        let bearing = Float(goalBearingDegrees * .pi / 180)
        try humanoid.setGoal(
            environment: 0,
            direction: F3(cos(bearing), sin(bearing), 0),
            distance: Float(goalDistance))
    }

    private func installSelectedGoal(in arachne: Arachne15LocomotionTask) throws {
        let bearing = Float(goalBearingDegrees * .pi / 180)
        try arachne.setGoal(
            environment: 0,
            direction: F3(cos(bearing), sin(bearing), 0),
            distance: Float(goalDistance))
    }

    func throwBox() {
        guard supportsBoxThrows else {
            interactionStatus = "this checkpoint scene has no projectile body"
            return
        }
        let state = isaacHumanoid?.environment.states()[0]
            ?? humanoid?.environment.states()[0]
        guard let state else { return }
        let forward: F3
        if let isaacHumanoid, isaacHumanoid.usesPointGoal {
            forward = isaacHumanoid.currentGoalDirection(environment: 0)
        } else if let humanoid, humanoid.usesPointGoal {
            forward = humanoid.currentGoalDirection(environment: 0)
        } else {
            forward = F3(1, 0, 0)
        }
        let lateral = F3(-forward.y, forward.x, 0)
        let target = state.torso.position
        let launchDistance: Float = 1.2
        let launch = target + lateral * (launchDistance * nextProjectileSide)
            + forward * 0.15
        let flightTime = launchDistance / Float(boxSpeed)
        let gravityValue = isaacHumanoid?.environment.scene.settings.gravity
            ?? humanoid?.environment.scene.settings.gravity ?? -9.81
        let gravity = F3(0, 0, gravityValue)
        let predictedTarget = target + state.torso.linearVelocity * flightTime
        let velocity = (predictedTarget - launch
            - 0.5 * gravity * flightTime * flightTime) / flightTime
        let angularVelocity = F3(
            nextProjectileSide * 2.5, -nextProjectileSide * 1.5,
            nextProjectileSide * 3.5)
        if let isaacHumanoid {
            isaacHumanoid.environment.throwProjectiles(
                environmentIDs: [0], positions: [launch],
                velocities: [velocity], angularVelocities: [angularVelocity])
        } else if let humanoid {
            humanoid.environment.throwProjectiles(
                environmentIDs: [0], positions: [launch],
                velocities: [velocity], angularVelocities: [angularVelocity])
        }
        let side = nextProjectileSide > 0 ? "left" : "right"
        let mass = isaacHumanoid?.environment.projectileMass
            ?? humanoid?.environment.projectileMass ?? 0
        interactionStatus = String(
            format: "%.1f kg physical box thrown from robot's %@ at %.1f m/s",
            mass, side, boxSpeed)
        nextProjectileSide *= -1
    }

    /// Polling is intentionally snapshot based. The mutable run root is never
    /// opened while the trainer may be replacing its individual files.
    func pollForLatestCheckpoint(force: Bool = false) {
        refreshTrainingMetrics()
        guard usesCheckpoint else { return }
        guard let liveRunDirectory, let task else { return }
        guard let candidate = VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: liveRunDirectory, task: task.spec.id,
                taskRevision: task.spec.revision) else {
            if actionProvider == nil {
                policyStatus = "waiting for first complete checkpoint in \(liveRunDirectory)"
            }
            return
        }
        newestUpdate = candidate.completedUpdates
        if actionProvider == nil,
           let metadata = replayMetadata(
                at: candidate.directory, task: task.spec.id),
           metadata.taskConfiguration != task.spec.configurationValues
                || metadata.maxEpisodeSteps != task.spec.maxEpisodeSteps
                || metadata.controlDecimation != task.spec.controlDecimation {
            // The viewer may launch before a new trainer has written its
            // metadata. Rebuild once from the first atomic checkpoint rather
            // than rejecting later checkpoints against provisional default
            // task options or structural horizon settings.
            rebuild()
            return
        }
        guard force || (autoLoadLatest
            && candidate.directory != loadedCheckpointDirectory
            && candidate.completedUpdates > (loadedUpdate ?? -1)) else { return }
        do {
            try installCheckpoint(candidate.directory, candidate: candidate)
            try restartEpisode()
        } catch {
            policyStatus = "new checkpoint rejected; replay kept previous policy: "
                + error.localizedDescription
        }
    }

    private func installCheckpoint(
        _ path: String, candidate: VectorPolicyCheckpointCandidate?
    ) throws {
        guard let task else { return }
        let loaded = try VectorPolicyRunner(checkpointDirectory: path)
        let mismatches = loaded.metadata.compatibilityMismatches(with: task.spec)
        guard mismatches.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint/task mismatch: \(mismatches.joined(separator: "; "))")
        }
        actionProvider = loaded
        loadedCheckpointDirectory = path
        if let candidate {
            loadedUpdate = candidate.completedUpdates
            policyStatus = "loaded complete update \(candidate.completedUpdates)  ·  "
                + "\(candidate.environmentSteps.formatted()) environment steps\n\(path)"
        } else {
            loadedUpdate = trainingState(at: path)?.completedUpdates
            policyStatus = "loaded \(path)/policy.safetensors"
        }
    }

    private func restartEpisode() throws {
        if robot == .unitreeH1 {
            guard let loadedCheckpointDirectory else {
                throw RLEnvironmentError.invalidConfiguration(
                    "Unitree H1 checkpoint directory is unavailable")
            }
            let session = try UnitreeH1Sim2SimSession(
                policyDirectory: loadedCheckpointDirectory)
            unitreeH1 = session
            unitreeInitialPosition = session.environment.state().root.position
            completed = 0; successes = 0; controlSteps = 0; accumulator = 0
            episodeFinished = false; running = true
            updateCameraTargets()
            applyCameraPreset()
            cameraEpoch += 1
            lastTime = CACurrentMediaTime()
            refreshStats()
            return
        }
        guard let task else { return }
        observation = try task.reset(seed: replaySeed)
        if let observation, let actionProvider {
            try actionProvider.reset(for: task, observation: observation)
        }
        if let arachne, arachne.usesPointGoal {
            let direction = arachne.currentGoalDirection(environment: 0)
            goalBearingDegrees = Double(
                atan2(direction.y, direction.x) * 180 / .pi)
            goalDistance = Double(
                arachne.currentGoalDistance(environment: 0))
        }
        result = RLStepBatch(spec: task.spec)
        completed = 0; successes = 0; controlSteps = 0; accumulator = 0
        episodeFinished = false
        running = actionProvider != nil
        updateCameraTargets()
        applyCameraPreset()
        cameraEpoch += 1
        lastTime = CACurrentMediaTime()
        refreshStats()
    }

    private func trainingState(at path: String) -> VectorPPOTrainingState? {
        try? JSONDecoder().decode(
            VectorPPOTrainingState.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(path)/training-state.json")))
    }

    private func refreshTrainingMetrics() {
        if !usesCheckpoint {
            trainingStatus = "deterministic classical controller · training not required"
            return
        }
        guard let liveRunDirectory else {
            trainingStatus = "static checkpoint replay"
            return
        }
        let url = URL(fileURLWithPath: liveRunDirectory)
            .appendingPathComponent("metrics.jsonl")
        guard let data = try? Data(contentsOf: url),
              let lastLine = data.split(separator: 0x0A).last,
              let metrics = try? JSONDecoder().decode(
                PPOUpdateMetrics.self, from: Data(lastLine)) else {
            trainingStatus = "trainer active · waiting for first PPO metric"
            return
        }
        let modificationDate = (try? FileManager.default.attributesOfItem(
            atPath: url.path)[.modificationDate]) as? Date
        let hasRecentHeartbeat = modificationDate.map {
            Date().timeIntervalSince($0) < 5
        } ?? false
        trainingStatus = String(
            format: "%@  ·  update %d  ·  %.0f steps/s\n"
                + "recent rollout success %.1f%%  ·  distance %.2f m  ·  length %.0f",
            hasRecentHeartbeat ? "trainer active" : "trainer idle",
            metrics.update + 1, metrics.stepsPerSecond,
            metrics.successRate * 100, metrics.meanEpisodeForwardDistance,
            metrics.meanEpisodeLength)
    }

    func tickIfRunning() {
        if unitreeH1 != nil {
            tickUnitreeH1IfRunning()
            return
        }
        guard running, let task, var observation, var result else { return }
        let now = CACurrentMediaTime()
        let wallStep = Double(task.spec.controlStep) / max(playbackRate, 0.1)
        if frameLockedCapture {
            accumulator = wallStep
        } else {
            accumulator += min(now - lastTime, 0.1)
        }
        lastTime = now
        var ticks = 0
        let maximumTicks = frameLockedCapture ? 1 : 3
        // Catch up to real time when synchronized Metal rendering is slower
        // than the 50 Hz control loop. The 0.1 s accumulator cap limits this
        // to three steps per draw, keeping UI input latency bounded.
        while accumulator >= wallStep, ticks < maximumTicks {
            do {
                let rootVelocityBeforeStep = isaacHumanoid?
                    .environment.states()[0].root.linearVelocity
                    ?? humanoid?.environment.states()[0].root.linearVelocity
                guard let actionProvider else {
                    running = false
                    return
                }
                let actions = try actionProvider.actions(
                    for: observation, task: task)
                try task.step(actions: actions, into: &result)
                try actionProvider.resetAfterStep(for: task, result: result)
                if (result.metrics["state/projectile_robot_contact"]?[0] ?? 0) > 0,
                   let before = rootVelocityBeforeStep {
                    let after = isaacHumanoid?
                        .environment.states()[0].root.linearVelocity
                        ?? humanoid?.environment.states()[0].root.linearVelocity
                        ?? before
                    interactionStatus = String(
                        format: "physical contact registered · root Δv %.3f m/s",
                        simd_length(after - before))
                }
                observation = result.observations
                controlSteps += 1
                updateCameraTargets()
                if result.terminated[0] || result.truncated[0] {
                    completed += 1
                    if result.successes[0] { successes += 1 }
                    running = false
                    episodeFinished = true
                    accumulator = 0
                    refreshStats()
                    break
                }
            } catch {
                policyStatus = "step failed: \(error.localizedDescription)"
                running = false
            }
            accumulator -= wallStep
            ticks += 1
        }
        self.observation = observation
        self.result = result
        if controlSteps.isMultiple(of: 10) { refreshStats() }
    }

    private func tickUnitreeH1IfRunning() {
        guard running, let session = unitreeH1 else { return }
        let now = CACurrentMediaTime()
        let wallStep = 0.02 / max(playbackRate, 0.1)
        if frameLockedCapture {
            accumulator = wallStep
        } else {
            accumulator += min(now - lastTime, 0.1)
        }
        lastTime = now
        var ticks = 0
        let maximumTicks = frameLockedCapture ? 1 : 3
        while accumulator >= wallStep, ticks < maximumTicks {
            do {
                _ = try session.step()
                controlSteps += 1
                updateCameraTargets()
                if controlSteps >= 500 {
                    running = false
                    episodeFinished = true
                    completed = 1
                    accumulator = 0
                    refreshStats()
                    break
                }
            } catch {
                policyStatus = "step failed: \(error.localizedDescription)"
                running = false
            }
            accumulator -= wallStep
            ticks += 1
        }
        if controlSteps.isMultiple(of: 10) { refreshStats() }
    }

    func singleStep() {
        guard !episodeFinished else { return }
        let wasRunning = running
        running = true
        // Do not fold wall time spent paused into a manual step. Without this
        // reset the 0.1-second catch-up allowance executes three control
        // transitions, making the button visibly skip frames.
        lastTime = CACurrentMediaTime()
        let controlStep = robot == .unitreeH1
            ? 0.02 : Double(task?.spec.controlStep ?? 1 / 30)
        accumulator = controlStep / max(playbackRate, 0.1)
        tickIfRunning()
        running = wasRunning
        refreshStats()
    }

    private func updateCameraTargets() {
        if let unitreeH1 {
            let state = unitreeH1.environment.state()
            replayCameraTarget = state.root.position + F3(0.15, 0, 0.15)
            courseCameraTarget = unitreeInitialPosition + F3(2, 0, 0.9)
        } else if let isaacHumanoid {
            let state = isaacHumanoid.environment.states()[0]
            replayCameraTarget = state.root.position + F3(0.15, 0, 0.15)
            let origin = isaacHumanoid.environment.refs[0].center
            let projection = isaacHumanoid.currentCommandProjection(environment: 0)
            courseCameraTarget = 0.5 * (origin + projection)
                + F3(0, 0, 0.9)
        } else if let humanoid {
            let state = humanoid.environment.states()[0]
            replayCameraTarget = state.root.position + F3(0.15, 0, 0.15)
            if humanoid.usesPointGoal {
                courseCameraTarget = 0.5 * (
                    humanoid.environment.refs[0].center
                        + humanoid.currentGoalPosition(environment: 0))
                    + F3(0, 0, 0.9)
            } else {
                let commandTarget = humanoid.currentCommandSpeed(environment: 0)
                    * Float(humanoid.spec.maxEpisodeSteps) * humanoid.spec.controlStep
                courseCameraTarget = humanoid.environment.refs[0].center
                    + F3(commandTarget * 0.5, 0, 0.9)
            }
        } else if let arm {
            let state = arm.environment.states()[0]
            let goal = arm.environment.refs[0].goalPosition
            replayCameraTarget = F3(
                0.5 * (state.blockPosition.x + goal.x),
                0.5 * (state.blockPosition.y + goal.y), 0.25)
            courseCameraTarget = replayCameraTarget
        } else if let arachne {
            let state = arachne.environment.states()[0]
            replayCameraTarget = state.root.position + F3(0.04, 0, 0.02)
            if arachne.usesPointGoal {
                courseCameraTarget = 0.5 * (
                    state.root.position
                        + arachne.currentGoalPosition(environment: 0))
                    + F3(0, 0, 0.05)
                return
            }
            let command = arachne.currentCommand(environment: 0)
            let heading = state.root.rotation.act(F3(1, 0, 0))
            let planarLength = sqrt(
                heading.x * heading.x + heading.y * heading.y)
            let planarHeading = planarLength > 1e-6
                ? F3(heading.x / planarLength, heading.y / planarLength, 0)
                : F3(1, 0, 0)
            courseCameraTarget = state.root.position
                + planarHeading * max(command.x, 0.05) * 2
                + F3(0, 0, 0.02)
        }
    }

    /// Install the selected preset in the solver settings that Renderer reads
    /// after `cameraEpoch` changes. Doing this only in SwiftUI's
    /// `updateNSView` loses a race with Renderer's scene-camera reset on the
    /// first draw and leaves the robot course-sized even though Follow Robot
    /// is selected.
    private func applyCameraPreset() {
        guard let solver else { return }
        let selectedTarget = cameraMode == .follow
            ? replayCameraTarget : courseCameraTarget
        let isArachne = arachne != nil
        solver.settings.cameraDistance = cameraMode == .follow
            ? (isArachne ? 0.65 : 3.2)
            : (isArachne ? 1.5 : 15)
        solver.settings.cameraTargetX = selectedTarget.x
        solver.settings.cameraTargetY = selectedTarget.y
        solver.settings.cameraTargetZ = selectedTarget.z
        solver.settings.cameraAzimuth = -.pi / 2
        solver.settings.cameraElevation = cameraMode == .follow ? 0.12 : 0.16
    }

    private func refreshStats() {
        if let unitreeH1 {
            let state = unitreeH1.environment.state()
            let displacement = state.root.position - unitreeInitialPosition
            let duration = max(Float(controlSteps) * 0.02, 0.02)
            let upright = state.root.rotation.act(F3(0, 0, 1)).z
            statsText = String(
                format: "forward %+.3f m   mean %.3f / command %.3f m/s\n"
                    + "lateral %+.3f m   height %.3f m   upright %.4f\n"
                    + "frame %d/500   time %.2f/10.00 s",
                displacement.x, displacement.x / duration, 0.5,
                displacement.y, state.root.position.z, upright,
                controlSteps, duration)
        } else if let arachne {
            let state = arachne.environment.states()[0]
            let command = arachne.currentCommand(environment: 0)
            let inverse = state.root.rotation.conjugate
            let velocity = inverse.act(state.root.linearVelocity)
            let angular = inverse.act(state.root.angularVelocity)
            let up = inverse.act(F3(0, 0, 1)).z
            if arachne.usesPointGoal {
                statsText = String(
                    format: "goal remaining %.3f m   planar speed %.3f m/s\n"
                        + "local velocity (%+.3f, %+.3f) / (%+.3f, %+.3f) m/s\n"
                        + "yaw rate %+.3f / %+.3f rad/s   up %.3f\n"
                        + "height %.3f m   frame %d/%d   success %d",
                    arachne.currentGoalDistance(environment: 0),
                    sqrt(state.root.linearVelocity.x
                        * state.root.linearVelocity.x
                        + state.root.linearVelocity.y
                            * state.root.linearVelocity.y),
                    velocity.x, velocity.y, command.x, command.y,
                    angular.z, command.z, up, state.root.position.z,
                    controlSteps, arachne.spec.maxEpisodeSteps, successes)
            } else {
                statsText = String(
                    format: "local velocity (%+.3f, %+.3f) / (%+.3f, %+.3f) m/s\n"
                        + "yaw rate %+.3f / %+.3f rad/s   up %.3f\n"
                        + "height %.3f m   frame %d/%d   episodes %d   success %d",
                    velocity.x, velocity.y, command.x, command.y,
                    angular.z, command.z, up, state.root.position.z,
                    controlSteps, arachne.spec.maxEpisodeSteps,
                    completed, successes)
            }
        } else if let isaacHumanoid {
            let state = isaacHumanoid.environment.states()[0]
            let command = isaacHumanoid.currentCommand(environment: 0)
            let forward3D = state.root.rotation.act(F3(1, 0, 0))
            let forwardLength = max(
                sqrt(forward3D.x * forward3D.x + forward3D.y * forward3D.y),
                1e-6)
            let heading = F3(
                forward3D.x / forwardLength, forward3D.y / forwardLength, 0)
            let lateral = F3(-heading.y, heading.x, 0)
            let measuredForward = simd_dot(state.root.linearVelocity, heading)
            let measuredLateral = simd_dot(state.root.linearVelocity, lateral)
            let projection = isaacHumanoid.currentCommandProjection(environment: 0)
            let projectionDelta = projection - state.root.position
            let projectionDistance = simd_length(F3(
                projectionDelta.x, projectionDelta.y, 0))
            let center = isaacHumanoid.environment.refs[0].center
            let displacement = state.root.position - center
            if isaacHumanoid.usesPointGoal {
                statsText = String(
                    format: "goal remaining %.3f m   speed %.3f / %.3f m/s\n"
                        + "yaw rate %+.3f / %+.3f rad/s   height %.3f m\n"
                        + "frame %d/%d   episodes %d   success %d",
                    projectionDistance,
                    sqrt(state.root.linearVelocity.x
                        * state.root.linearVelocity.x
                        + state.root.linearVelocity.y
                            * state.root.linearVelocity.y),
                    command.x, state.root.angularVelocity.z, command.z,
                    state.root.position.z, controlSteps,
                    isaacHumanoid.spec.maxEpisodeSteps, completed, successes)
            } else {
                statsText = String(
                    format: "local speed %+.3f / %.3f m/s   lateral %+.3f m/s\n"
                        + "yaw rate %+.3f / %+.3f rad/s   command projection %.2f m\n"
                        + "world Δ (%+.2f, %+.2f) m   height %.3f m\n"
                        + "frame %d/%d   episodes %d   success %d",
                    measuredForward, command.x, measuredLateral,
                    state.root.angularVelocity.z, command.z, projectionDistance,
                    displacement.x, displacement.y, state.root.position.z,
                    controlSteps, isaacHumanoid.spec.maxEpisodeSteps,
                    completed, successes)
            }
        } else if let humanoid {
            let s = humanoid.environment.states()[0]
            let forward = s.torso.rotation.act(F3(1, 0, 0))
            let headingMagnitude = max(
                sqrt(forward.x * forward.x + forward.y * forward.y), 1e-6)
            let headingAlignment = forward.x / headingMagnitude
            let command = humanoid.currentCommandSpeed(environment: 0)
            let measuredVelocity = humanoid.currentMeasuredRootVelocity(environment: 0)
            if humanoid.usesPointGoal {
                let toGoal = humanoid.currentGoalPosition(environment: 0)
                    - s.root.position
                let distance = simd_length(F3(toGoal.x, toGoal.y, 0))
                let direction = distance > 1e-6
                    ? F3(toGoal.x, toGoal.y, 0) / distance : F3(1, 0, 0)
                let goalHeading = simd_dot(
                    F3(forward.x, forward.y, 0) / Float(headingMagnitude), direction)
                statsText = String(
                    format: "goal remaining %.3f m   speed %.3f / %.3f m/s\n"
                        + "goal heading %.3f   foot exchanges %d\n"
                        + "height %.3f m   frame %d/%d   episodes %d   success %d",
                    distance, simd_length(F3(
                        measuredVelocity.x, measuredVelocity.y, 0)), command,
                    goalHeading, humanoid.currentAlternatingSteps(environment: 0),
                    s.root.position.z, controlSteps, humanoid.spec.maxEpisodeSteps,
                    completed, successes)
            } else {
                let target = command * Float(humanoid.spec.maxEpisodeSteps)
                    * humanoid.spec.controlStep
                statsText = String(format: "A→B x %+.3f / %.3f m   speed %+.3f / %.3f m/s\nheading %.3f   lateral %+.3f m   forward foot exchanges %d\nheight %.3f m   frame %d/%d   episodes %d   success %d",
                                   s.root.position.x - humanoid.environment.refs[0].center.x,
                                   target, measuredVelocity.x, command,
                                   headingAlignment,
                                   s.root.position.y - humanoid.environment.refs[0].center.y,
                                   humanoid.currentAlternatingSteps(environment: 0),
                                   s.root.position.z,
                                   controlSteps, humanoid.spec.maxEpisodeSteps,
                                   completed, successes)
            }
        } else if let arm {
            let s = arm.environment.states()[0], goal = arm.environment.refs[0].goalPosition
            statsText = String(format: "T (%.2f, %.2f)   goal (%.2f, %.2f)\ndistance %.3f m   yaw error %.3f rad\nstep %d/%d   episodes %d   success %d",
                               s.blockPosition.x, s.blockPosition.y, goal.x, goal.y,
                               simd_length(goal - s.blockPosition), abs(s.blockYaw),
                               controlSteps, arm.spec.maxEpisodeSteps,
                               completed, successes)
        }
    }
}

struct PolicyReplayLabView: View {
    @StateObject private var model = PolicyReplayModel()
    private let checkpointTimer = Timer.publish(
        every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        HSplitView {
            PolicyReplayMetalView(model: model).frame(minWidth: 650)
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Policy Replay").font(.title2).bold()
                    Text(model.scenarioSummary)
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Picker("Task", selection: $model.robot) {
                        ForEach(PolicyReplayModel.Robot.allCases, id: \.self) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                    .pickerStyle(.menu)
                    HStack {
                        Button(model.episodeFinished ? "Replay" : (model.running ? "Pause" : "Play")) {
                            model.togglePlayback()
                        }
                        Button("Step") { model.singleStep() }
                            .disabled(model.episodeFinished)
                        Button("Reset") { model.resetEpisode() }
                        Button("Load Latest") {
                            model.pollForLatestCheckpoint(force: true)
                        }
                        .disabled(!model.usesCheckpoint)
                    }
                    Toggle("Auto-load complete checkpoints",
                           isOn: $model.autoLoadLatest)
                        .disabled(!model.usesCheckpoint)
                    Picker("Camera", selection: $model.cameraMode) {
                        ForEach(PolicyReplayModel.CameraMode.allCases, id: \.self) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                    .pickerStyle(.segmented)
                    HStack {
                        Text("Playback").font(.caption)
                        Slider(value: $model.playbackRate, in: 0.25...1)
                        Text(String(format: "%.2fx", model.playbackRate))
                            .font(.caption.monospacedDigit()).frame(width: 42)
                    }
                    Text(model.statsText)
                        .font(.system(.caption, design: .monospaced))
                        .fixedSize(horizontal: false, vertical: true)
                    if model.supportsGoalPlacement {
                        Divider()
                        Text("Interactive goal").font(.headline)
                        HStack {
                            Text("Bearing").font(.caption).frame(width: 50, alignment: .leading)
                            Slider(value: $model.goalBearingDegrees, in: -180...180,
                                   step: 5)
                            Text(String(format: "%+.0f°", model.goalBearingDegrees))
                                .font(.caption.monospacedDigit()).frame(width: 42)
                        }
                        HStack {
                            Text("Distance").font(.caption).frame(width: 50, alignment: .leading)
                            Slider(value: $model.goalDistance,
                                   in: model.goalDistanceRange,
                                   step: model.robot == .arachneGoal
                                    || model.robot == .arachneClassical
                                        ? 0.1 : 0.5)
                            Text(String(format: "%.1f m", model.goalDistance))
                                .font(.caption.monospacedDigit()).frame(width: 42)
                        }
                        HStack {
                            Button("Apply & Reset") { model.applyGoalAndReset() }
                            Button("Sample Task Goal") { model.randomizeGoalAndReset() }
                        }
                        Text(model.goalControlExplanation)
                            .font(.caption2).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Divider()
                        Text("Impact test").font(.headline)
                        Text(model.impactModelSummary)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                        HStack {
                            Text("Speed").font(.caption).frame(width: 50, alignment: .leading)
                            Slider(value: $model.boxSpeed, in: 2...10, step: 0.5)
                            Text(String(format: "%.1f", model.boxSpeed))
                                .font(.caption.monospacedDigit()).frame(width: 42)
                        }
                        Button("Throw Physical Box") { model.throwBox() }
                            .disabled(!model.supportsBoxThrows || model.episodeFinished)
                        if !model.supportsBoxThrows {
                            Text("Load a goal checkpoint trained with projectiles to enable its colliding box body.")
                                .font(.caption2).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        if !model.interactionStatus.isEmpty {
                            Text(model.interactionStatus)
                                .font(.caption2.monospaced()).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    Divider()
                    Text(model.controllerSectionTitle).font(.headline)
                    Text(model.trainingStatus)
                        .font(.system(.caption, design: .monospaced))
                        .fixedSize(horizontal: false, vertical: true)
                    if let loaded = model.loadedUpdate, let newest = model.newestUpdate {
                        Text("visible checkpoint \(loaded)  ·  newest complete \(newest)")
                            .font(.caption.monospacedDigit())
                    }
                    Text(model.policyStatus).font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Drag to orbit, right-drag to pan, and scroll to zoom. Follow Robot keeps footfalls large enough to inspect; Show Course keeps the route in view.")
                        .font(.caption2).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(model.usesCheckpoint
                        ? "This view executes only the learned Safetensors checkpoint used by `eval-rl`; no scripted or reference controller is available in learned replay."
                        : "This baseline contains no neural inference or checkpoint. Its motion is generated by the displayed CPG/IK controller through the exact same physical plant used by RL.")
                        .font(.caption2).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(14)
            }
            .frame(minWidth: 520, idealWidth: 600, maxWidth: 720)
        }
        .onReceive(checkpointTimer) { _ in
            model.pollForLatestCheckpoint()
        }
    }
}

private struct PolicyReplayMetalView: NSViewRepresentable {
    @ObservedObject var model: PolicyReplayModel
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> PolicyOrbitMTKView {
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()!
        let view = PolicyOrbitMTKView(frame: .zero, device: device)
        if ProcessInfo.processInfo.environment["AVBD_SHOT"] != nil
            || ProcessInfo.processInfo.environment["AVBD_VIDEO_DIR"] != nil {
            view.framebufferOnly = false
        }
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = Renderer.sampleCount
        view.preferredFramesPerSecond = 30
        let renderer = try! Renderer(device: device, model: model)
        context.coordinator.renderer = renderer
        view.renderer = renderer
        view.delegate = renderer
        return view
    }
    func updateNSView(_ view: PolicyOrbitMTKView, context: Context) {
        guard let renderer = context.coordinator.renderer else { return }
        if context.coordinator.cameraMode != model.cameraMode {
            context.coordinator.cameraMode = model.cameraMode
            renderer.azimuth = -.pi / 2
            renderer.elevation = model.cameraMode == .follow ? 0.12 : 0.16
            let isArachne = model.robot == .arachne
                || model.robot == .arachneGoal
                || model.robot == .arachneClassical
            renderer.distance = model.cameraMode == .follow
                ? (isArachne ? 0.65 : 3.2)
                : (isArachne ? 1.5 : 15)
        }
        renderer.target = model.cameraMode == .follow
            ? model.replayCameraTarget : model.courseCameraTarget
    }
    @MainActor final class Coordinator {
        var renderer: Renderer?
        var cameraMode: PolicyReplayModel.CameraMode?
    }
}

/// Camera interaction local to Policy Replay. It intentionally does not
/// expose physics dragging: learned actions remain the sole robot controls,
/// while left-drag, right-drag, and scroll manipulate only the view.
private final class PolicyOrbitMTKView: MTKView {
    weak var renderer: Renderer?
    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        guard let renderer else { return }
        renderer.azimuth -= Float(event.deltaX) * 0.008
        renderer.elevation = min(max(
            renderer.elevation + Float(event.deltaY) * 0.008, -1.5), 1.55)
    }

    override func rightMouseDragged(with event: NSEvent) {
        guard let renderer else { return }
        let scale = renderer.distance * 0.0015
        let right = F3(-sin(renderer.azimuth), cos(renderer.azimuth), 0)
        renderer.target -= right * Float(event.deltaX) * scale
        renderer.target += F3(0, 0, 1) * Float(event.deltaY) * scale
    }

    override func scrollWheel(with event: NSEvent) {
        guard let renderer else { return }
        renderer.distance = min(max(
            renderer.distance * (1 - Float(event.scrollingDeltaY) * 0.02),
            1.2), 100)
    }
}
