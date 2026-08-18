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
    private enum FlowFrontierQualification {
        case measuredFinitePrefix
        case reusableRecoverySafeFrontier

        var endpointLabel: String {
            switch self {
            case .measuredFinitePrefix:
                return "MEASURED FINITE-PREFIX ENDPOINT · NOT REUSABLE"
            case .reusableRecoverySafeFrontier:
                return "REUSABLE RECOVERY-SAFE FRONTIER"
            }
        }
    }

    enum Robot: CaseIterable {
        case unitreeH1
        case gearSonicG1
        case humanoidIsaac
        case humanoidIsaacGoal
        case humanoidBoxCarry
        case arachne
        case arachneGoal
        case arachneClassical

        /// Only current catalog entries plus the explicit local box-carry
        /// development surface belong in the picker. Historical learned
        /// checkpoints remain addressable for a visible compatibility error,
        /// but can never be selected accidentally.
        static let allCases: [Robot] = [
            .unitreeH1, .gearSonicG1, .humanoidIsaac,
            .humanoidBoxCarry, .arachneClassical,
        ]

        var selectionID: String {
            switch self {
            case .unitreeH1: return "unitree-h1-sim2sim-v0"
            case .gearSonicG1: return "gear-sonic-g1-reference-v0"
            case .humanoidIsaac: return "humanoid-isaac-flat-v2"
            case .humanoidIsaacGoal: return "humanoid-isaac-goal-v0"
            case .humanoidBoxCarry: return "humanoid-box-carry-v0"
            case .arachne: return "arachne15-velocity-v0"
            case .arachneGoal: return "arachne15-goal-v0"
            case .arachneClassical: return "arachne15-classical-goal-v0"
            }
        }

        private var catalogEntry: PolicyReplayCatalogEntry {
            // Hidden historical cases are used only to explain why an
            // explicitly requested old checkpoint is incompatible. They must
            // not be returned by allCases or normal selection lookup.
            if let entry = PolicyReplayCatalog.entry(selectionID: selectionID)
                ?? PolicyReplayCatalog.historicalEntry(
                    selectionID: selectionID) {
                return entry
            }
            preconditionFailure(
                "replay robot has no catalog declaration: \(selectionID)")
        }

        var displayName: String {
            self == .humanoidBoxCarry
                ? "H1 Box Carry (Development)" : catalogEntry.displayName
        }
        var taskID: String {
            self == .humanoidBoxCarry
                ? "humanoid-box-carry-v0" : catalogEntry.taskID
        }
        var runtime: PolicyReplayRuntime {
            self == .humanoidBoxCarry ? .nativeMLX : catalogEntry.runtime
        }

        var usesClassicalController: Bool {
            runtime == .classicalController
        }

        static func fromSelectionID(_ id: String) -> Robot? {
            // Migrate retired picker identifiers. This is intentionally
            // a UI preference migration, not a checkpoint compatibility
            // bypass: an explicit old checkpoint is still rejected by the
            // current task-revision check during installation.
            if id == "humanoid-isaac-flat-v0"
                || id == "humanoid-isaac-flat-v1" {
                return .humanoidIsaac
            }
            return allCases.first { $0.selectionID == id }
        }

        init?(taskID: String) {
            switch taskID {
            case "unitree-h1-sim2sim-v0": self = .unitreeH1
            case "gear-sonic-g1-reference-v0": self = .gearSonicG1
            case "humanoid-isaac-flat-v0": self = .humanoidIsaac
            case "humanoid-isaac-goal-v0": self = .humanoidIsaacGoal
            case "humanoid-box-carry-v0": self = .humanoidBoxCarry
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
        let selectionKey = "AVBDPolicyReplaySelectedTask"
        let persistedTask = UserDefaults.standard.string(forKey: selectionKey)
        let requestedTask = environment["AVBD_REPLAY_TASK"] ?? persistedTask
        let selected = requestedTask.flatMap(Robot.fromSelectionID)
        if environment["AVBD_REPLAY_TASK"] == nil,
           persistedTask.map(
               ["humanoid-isaac-flat-v0", "humanoid-isaac-flat-v1"]
                   .contains) == true,
           let selected {
            UserDefaults.standard.set(
                selected.selectionID, forKey: selectionKey)
        }
        return selected ?? .humanoidIsaac
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
    @Published var playbackRate: Double = ProcessInfo.processInfo.environment[
        "AVBD_REPLAY_RATE"].flatMap(Double.init) ?? 1
    @Published var statsText = ""
    @Published var policyStatus = ""
    @Published var trainingStatus = "waiting for trainer metrics"
    @Published var interactionStatus = ""
    @Published var goalBearingDegrees: Double = 0
    @Published var goalDistance: Double = 8
    @Published var boxSpeed: Double = 6
    @Published private(set) var gearSonicReferenceNames: [String] = []
    @Published private(set) var selectedGEARSonicReference = ""
    @Published var autoLoadLatest: Bool = {
        guard let value = ProcessInfo.processInfo.environment[
            "AVBD_REPLAY_AUTO_LOAD_LATEST"]?.lowercased() else { return true }
        return !["0", "false", "no", "off"].contains(value)
    }()
    @Published private(set) var loadedUpdate: Int?
    @Published private(set) var newestUpdate: Int?
    @Published private(set) var episodeFinished = false
    @Published private(set) var arachneFolded = false
    @Published private(set) var replayCameraTarget = F3(0, 0, 1)
    @Published private(set) var courseCameraTarget = F3(6.5, 0, 0.9)
    var colorByGraphColor = false
    private(set) var cameraEpoch = 0

    private var isaacHumanoid: HumanoidIsaacVelocityTask?
    private var humanoidBoxCarry: HumanoidBoxCarryTask?
    private var arachne: Arachne15LocomotionTask?
    private var unitreeH1: UnitreeH1Sim2SimSession?
    private var gearSonicG1: GEARSonicG1Session?
    private var task: (any VectorizedRLTask)?
    private var observation: RLObservationBatch?
    private var result: RLStepBatch?
    private var actionProvider: (any RLActionProvider)?
    private var flowReplayController:
        HumanoidBoxFlowDistillation.ReplayController?
    private var flowFrontierQualification:
        FlowFrontierQualification?
    private var flowReplayUsesLearnedActionChunk = false
    private var arachneRevealController: Arachne15RevealController?
    private var arachneAutoUnfold = false
    private var loadedCheckpointDirectory: String?
    private var explicitCheckpointPath: String? {
        ProcessInfo.processInfo.environment["AVBD_REPLAY_CHECKPOINT"]
    }
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
    /// Deterministic visual-regression hook. It invokes the exact same model
    /// action as the UI button; normal interactive runs leave it unset.
    private let scriptedBoxThrowStep = ProcessInfo.processInfo.environment[
        "AVBD_REPLAY_AUTO_THROW_STEP"].flatMap(Int.init)
    private var scriptedBoxThrowPerformed = false
    private var controlSteps = 0
    private var completed = 0
    private var successes = 0
    private var nextProjectileSide: Float = 1
    private var replaySeed: UInt64 = 21_001
    private var unitreeInitialPosition = F3.zero
    private var gearSonicInitialPosition = F3.zero
    private var gearSonicCourseCenter = F3(0, 0, 0.8)
    private var gearSonicCourseDistance: Float = 4.5
    private var gearSonicBundleDirectory: String?
    private var gearSonicReferenceDirectory: String?
    private var gearSonicReferenceDirectories: [String: String] = [:]

    private static func flowFrontierQualification(
        reportPath: String
    ) throws -> FlowFrontierQualification {
        let data = try Data(contentsOf: URL(fileURLWithPath: reportPath))
        guard let dictionary = try JSONSerialization.jsonObject(with: data)
                as? [String: Any] else {
            throw RLEnvironmentError.invalidConfiguration(
                "flow replay report is not a JSON object")
        }
        let finitePrefixPassed = (dictionary[
            "targetFinitePrefixGatePassed"] as? Bool)
            ?? (dictionary["targetPlanningGatePassed"] as? Bool)
            ?? ((dictionary["targetGatePassed"] as? Bool) == true
                && ((dictionary["targetCloneSuccessFraction"]
                    as? NSNumber)?.floatValue ?? 0) >= 0.8
                && ((dictionary["targetReplayMaximumNormalizedError"]
                    as? NSNumber)?.floatValue ?? .infinity) < 0.02)
        let recoveryPathSafe = (dictionary[
            "targetPredictedRecoveryPathSafe"] as? Bool)
            ?? (dictionary["recedingHorizonSteps"] == nil)
        let reusableFrontierPassed = (dictionary[
            "targetReusableFrontierGatePassed"] as? Bool)
            ?? (finitePrefixPassed && recoveryPathSafe)
        return reusableFrontierPassed
            ? .reusableRecoverySafeFrontier : .measuredFinitePrefix
    }

    var supportsGoalPlacement: Bool {
        isaacHumanoid?.usesPointGoal == true
            || arachne?.usesPointGoal == true
    }
    var supportsArachneReveal: Bool { arachne != nil }
    var hasArachneTransformation: Bool {
        arachneRevealController != nil
    }
    var isArachneTransforming: Bool {
        arachneRevealController != nil && !arachneFolded
    }
    /// A learned actor is active, regardless of whether it came from native
    /// training or an immutable external import.
    var usesCheckpoint: Bool { !robot.usesClassicalController }
    var supportsLiveCheckpointLoading: Bool {
        robot.runtime == .nativeMLX && explicitCheckpointPath == nil
    }
    var supportsGEARSonicReferenceSelection: Bool { robot == .gearSonicG1 }
    var hasReplayScene: Bool { solver != nil }
    var controllerSectionTitle: String {
        switch robot.runtime {
        case .nativeMLX: return "Parallel training"
        case .unitreeRecurrentMLX, .externalReferenceMLX:
            return "Imported policy"
        case .classicalController: return "Controller"
        }
    }
    var replayControlExplanation: String {
        switch robot.runtime {
        case .nativeMLX:
            if flowReplayController != nil {
                let controller = flowReplayUsesLearnedActionChunk
                    ? "learned MLX action chunk"
                    : "exact simulator-feedback teacher"
                if flowFrontierQualification
                        == .reusableRecoverySafeFrontier {
                    return "This experiment replays its exact measured physical source lineage, then executes the \(controller). The app pauses at the recovery-tested frontier; this boundary is qualified for a subsequent controller."
                }
                return "This experiment replays its exact measured physical source lineage, then executes the \(controller) for its finite evaluated horizon. The app pauses at that endpoint; recovery was not established, so it is not a reusable continuation boundary."
            }
            return "Normal motion executes the learned Safetensors checkpoint used by `eval-rl`. Arachne transformation controls temporarily command a bounded commissioning trajectory through the same motors and physics; learned control resumes only after deployment settles."
        case .unitreeRecurrentMLX:
            return "This is a static, source-verified public recurrent policy imported to MLX. It is not connected to the native trainer or checkpoint hot-reload path."
        case .externalReferenceMLX:
            return "This is a static, source-verified GEAR-SONIC policy imported from ONNX to native MLX. The selected reference clip supplies commands; every rendered link still moves only through the policy, motors, and physical contacts."
        case .classicalController:
            return "This baseline contains no neural inference or checkpoint. Its motion is generated by the displayed CPG/IK controller through the exact same physical plant used by RL."
        }
    }
    var goalControlExplanation: String {
        usesCheckpoint
            ? "Changing the goal updates the learned policy's command; it does not steer the joints directly."
            : "Changing the goal updates body-twist steering. The CPG and IK produce joint targets; only motor torque and physical contacts move the body."
    }
    var goalDistanceRange: ClosedRange<Double> {
        arachne?.usesPointGoal == true ? 0.4...3.0 : 2...12
    }
    var boxSpeedRange: ClosedRange<Double> {
        switch robot {
        case .arachne, .arachneGoal, .arachneClassical: return 0.5...4
        default: return 2...10
        }
    }
    var boxSpeedStep: Double {
        switch robot {
        case .arachne, .arachneGoal, .arachneClassical: return 0.25
        default: return 0.5
        }
    }
    var supportsBoxThrows: Bool {
        gearSonicG1 != nil
            || unitreeH1 != nil
            || isaacHumanoid?.hasProjectile(environment: 0) == true
            || arachne?.environment.hasProjectile(environment: 0) == true
    }
    var impactModelSummary: String {
        if let gearSonicG1 {
            return String(
                format: "%.1f kg box · all 29 G1 training colliders",
                gearSonicG1.environment.configuration.projectileMass)
        }
        if let isaacHumanoid, isaacHumanoid.hasProjectile(environment: 0) {
            return String(
                format: "%.1f kg box · full imported H1 collision primitives",
                isaacHumanoid.environment.projectileMass)
        }
        if let unitreeH1 {
            return String(
                format: "%.1f kg box · full source H1 collision plant",
                unitreeH1.environment.projectileMass)
        }
        if let arachne,
           arachne.environment.hasProjectile(environment: 0) {
            return String(
                format: "%.2f kg box · all 39 Arachne collision primitives",
                arachne.environment.projectileMass)
        }
        return "no physical projectile in this checkpoint scene"
    }
    var scenarioSummary: String {
        switch robot {
        case .unitreeH1:
            return "Unchanged public Unitree RL Gym H1 recurrent policy, imported from TorchScript to MLX and running on AVBD Metal physics."
        case .gearSonicG1:
            return "NVIDIA GEAR-SONIC's 29-DoF G1 reference-following policy, imported from ONNX to native batched MLX and replayed on its analytic training collision plant. Full walking and dance clips pass; high-dynamic clips remain development-qualified."
        case .humanoidIsaac:
            return "Accepted native MLX H1 flat locomotion: unchanged weights requalified on the current BSD-source collision hulls with zero target updates, passing 2,028 of 2,048 sealed episodes (99.02%) across four fixed seeds. Markers show the commanded velocity segment, not a point goal."
        case .humanoidIsaacGoal:
            return "Development H1 point-goal/8 kg impact policy on the current actuator contract (78.1% sealed-test goal success); PPO owns every joint action, but this policy is not acceptance-qualified yet."
        case .humanoidBoxCarry:
            if flowFrontierQualification == .measuredFinitePrefix {
                return "Measured finite-prefix H1 box experiment on the exact batched training scene. It replays the recorded physical controls only through the evaluated horizon. The endpoint is not recovery-qualified, so this is diagnostic evidence—not a solved transport skill or reusable frontier."
            }
            if flowReplayController?.isBalanceOnly == true {
                let controller = flowReplayUsesLearnedActionChunk
                    ? "the immutable MLX action chunk"
                    : "the exact simulator-feedback teacher"
                return "Recovery-safe loaded-balance frontier on the exact batched training scene: the H1 physically grasps the 2 kg box, replays the measured source lineage, then \(controller) owns the balance controls. This is not receiving-table transport or release."
            }
            if flowReplayController != nil {
                return "Reusable recovery-safe destination-progress frontier on the exact batched training scene: the H1 physically grasps the 2 kg box, replays the measured source lineage, then advances the held load toward the receiving table. Live goal error is the evidence; placement and release are not learned yet."
            }
            return "Development H1 carry policy on the exact batched training scene. Live metrics below describe what the selected checkpoint actually achieves; receiving-table placement and release are not learned yet."
        case .arachne:
            return "Accepted Arachne-15 straight-walk benchmark at 0.15 m/s with the exact printable CAD visuals and corrected revision-6 foot collision model."
        case .arachneGoal:
            return "Arachne-15 samples a random world target, converts it to the reusable local velocity/yaw command, and must enter the visible target slowly enough to stop there."
        case .arachneClassical:
            return "Non-neural paired-ripple CPG and exact two-axis leg IK drive the same torque-limited motors, contacts, randomized plant, and arbitrary point-goal task as the learned policy."
        }
    }

    var solver: GPUSolver? {
        if let gearSonicG1 { return gearSonicG1.environment.solver }
        if let unitreeH1 { return unitreeH1.environment.solver }
        if let humanoidBoxCarry { return humanoidBoxCarry.environment.solver }
        if let isaacHumanoid { return isaacHumanoid.environment.solver }
        return arachne?.environment.solver
    }

    func selectGEARSonicReference(_ name: String) {
        guard robot == .gearSonicG1,
              name != selectedGEARSonicReference,
              gearSonicReferenceDirectories[name] != nil else { return }
        selectedGEARSonicReference = name
        UserDefaults.standard.set(
            name, forKey: "AVBDPolicyReplayGEARSonicReference")
        rebuild()
    }

    func gearSonicReferenceDisplayName(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
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

    func reportRenderFailure(_ message: String) {
        running = false
        accumulator = 0
        let status = "render stopped: \(message)"
        // statsText is near the top of the controls while policyStatus keeps
        // the diagnosis visible in the controller section as well.
        statsText = status
        policyStatus = status
    }

    func rebuild() {
        unitreeH1 = nil; gearSonicG1 = nil; isaacHumanoid = nil
        humanoidBoxCarry = nil
        arachne = nil
        arachneRevealController = nil
        arachneFolded = false
        arachneAutoUnfold = false
        task = nil; actionProvider = nil; flowReplayController = nil
        flowFrontierQualification = nil
        flowReplayUsesLearnedActionChunk = false
        gearSonicBundleDirectory = nil
        gearSonicReferenceDirectory = nil
        gearSonicReferenceDirectories = [:]
        gearSonicReferenceNames = []
        selectedGEARSonicReference = ""
        loadedCheckpointDirectory = nil; loadedUpdate = nil; newestUpdate = nil
        completed = 0; successes = 0; controlSteps = 0; accumulator = 0
        replaySeed = 21_001
        scriptedBoxThrowPerformed = false
        boxSpeed = switch robot {
        case .arachne, .arachneGoal, .arachneClassical: 2.5
        default: 6
        }
        interactionStatus = ""
        episodeFinished = false
        running = true
        do {
            if robot == .gearSonicG1 {
                try installGEARSonicG1Replay()
                updateCameraTargets()
                applyCameraPreset()
                cameraEpoch += 1
                lastTime = CACurrentMediaTime()
                refreshStats()
                return
            }
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
            let overridePath = explicitCheckpointPath
            let checkpointRelativeDirectory: String
            if robot.usesClassicalController {
                checkpointRelativeDirectory = desiredTaskID
            } else if robot == .humanoidBoxCarry {
                // Development-only local preview. It deliberately does not
                // enter the curated replay catalog until an accepted policy
                // and its machine-readable evidence are packaged.
                checkpointRelativeDirectory = desiredTaskID
            } else {
                let declaredEntry = PolicyReplayCatalog.entry(
                    selectionID: robot.selectionID)
                    ?? (overridePath == nil ? nil
                        : PolicyReplayCatalog.historicalEntry(
                            selectionID: robot.selectionID))
                guard let declared = declaredEntry?
                    .checkpointRelativeDirectory else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "learned replay has no packaged checkpoint declaration")
                }
                checkpointRelativeDirectory = declared
            }
            let packagedPath = "checkpoints/\(checkpointRelativeDirectory)"
            let bundledPath = Bundle.main.resourceURL?
                .appendingPathComponent(
                    "checkpoints/\(checkpointRelativeDirectory)").path
            let configurationPaths = robot.usesClassicalController ? []
                : PolicyReplayCheckpointResolution.candidates(
                    explicit: overridePath, fallbacks: [
                        autoLoadLatest ? liveRunDirectory : nil,
                        bundledPath, packagedPath,
                    ])
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
                    includeInteractiveRobustnessProbes: true,
                    options: replayOptions))
            isaacHumanoid = configuredTask as? HumanoidIsaacVelocityTask
            humanoidBoxCarry = configuredTask as? HumanoidBoxCarryTask
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
            guard let task else { return }
            let liveCheckpoint: VectorPolicyCheckpointCandidate?
            if robot.usesClassicalController || overridePath != nil {
                liveCheckpoint = nil
            } else {
                liveCheckpoint = liveRunDirectory.flatMap {
                    VectorPolicyCheckpointDiscovery.latestCompleteCheckpoint(
                        inRunDirectory: $0, task: task.spec.id,
                        taskRevision: task.spec.revision)
                }
            }
            newestUpdate = liveCheckpoint?.completedUpdates
            let startupLiveCheckpoint = autoLoadLatest ? liveCheckpoint : nil
            let candidates = robot.usesClassicalController ? []
                : PolicyReplayCheckpointResolution.candidates(
                    explicit: overridePath, fallbacks: [
                        startupLiveCheckpoint?.directory, bundledPath,
                        packagedPath,
                    ])
            var loadFailures = [String]()
            if let overridePath,
               !FileManager.default.fileExists(
                    atPath: "\(overridePath)/metadata.json") {
                loadFailures.append(
                    "\(overridePath): metadata.json not found")
            }
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
            if robot == .humanoidBoxCarry,
               let flowReport = ProcessInfo.processInfo.environment[
                    "AVBD_REPLAY_FLOW_REPORT"],
               let sourceCheckpoint = loadedCheckpointDirectory {
                let actionChunk = ProcessInfo.processInfo.environment[
                    "AVBD_REPLAY_ACTION_CHUNK"]
                let controller = try HumanoidBoxFlowDistillation
                    .ReplayController(
                        checkpointDirectory: sourceCheckpoint,
                        flowReportPath: flowReport,
                        actionChunkCheckpointDirectory: actionChunk)
                flowReplayController = controller
                flowFrontierQualification = try Self
                    .flowFrontierQualification(reportPath: flowReport)
                flowReplayUsesLearnedActionChunk = actionChunk != nil
                let controllerName = actionChunk == nil
                    ? "exact simulator-feedback teacher"
                    : "learned MLX action chunk"
                let qualificationName = flowFrontierQualification
                        == .reusableRecoverySafeFrontier
                    ? "reusable recovery-safe frontier"
                    : "measured finite prefix (not reusable)"
                actionProvider = controller
                policyStatus = "loaded \(controllerName)\n\(flowReport)"
                trainingStatus = "\(qualificationName) · replay pauses at the measured endpoint"
            }
            if actionProvider == nil {
                running = false
                policyStatus = loadFailures.isEmpty
                    ? "no promoted checkpoint for \(task.spec.id)"
                    : "no compatible checkpoint: \(loadFailures.joined(separator: "; "))"
            }
            try restartEpisode()
            if ProcessInfo.processInfo.environment[
                    "AVBD_REPLAY_AUTO_REVEAL"] == "1",
               arachne != nil {
                startArachneFold(autoUnfold: true)
            }
            refreshTrainingMetrics()
        } catch {
            running = false
            accumulator = 0
            policyStatus = "replay unavailable: \(error.localizedDescription)"
            // State/camera/stat helpers use legacy synchronous readers. A
            // terminal solver must be rebuilt before any of them are legal.
            if solver?.runtimeFailure != nil {
                cameraEpoch += 1
                lastTime = CACurrentMediaTime()
                return
            }
        }
        updateCameraTargets()
        applyCameraPreset()
        cameraEpoch += 1
        lastTime = CACurrentMediaTime()
        refreshStats()
    }

    private func installUnitreeH1Replay() throws {
        let overridePath = explicitCheckpointPath
        let bundledPath = Bundle.main.resourceURL?
            .appendingPathComponent("checkpoints/external/unitree-h1").path
        let candidates = PolicyReplayCheckpointResolution.candidates(
            explicit: overridePath, fallbacks: [
                bundledPath, "checkpoints/external/unitree-h1",
            ])
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

    private func installGEARSonicG1Replay() throws {
        let environment = ProcessInfo.processInfo.environment
        let overridePath = environment["AVBD_GEAR_SONIC_BUNDLE"]
            ?? environment["AVBD_REPLAY_CHECKPOINT"]
        let bundledPath = Bundle.main.resourceURL?
            .appendingPathComponent(
                "checkpoints/external/gear-sonic-g1").path
        let candidates = PolicyReplayCheckpointResolution.candidates(
            explicit: overridePath, fallbacks: [
                bundledPath, "checkpoints/external/gear-sonic-g1",
            ])
        guard let bundleDirectory = candidates.first(where: {
            Self.isGEARSonicBundle(at: $0)
        }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "imported GEAR-SONIC G1 bundle not found; run "
                    + "Tools/import_gear_sonic_policy.py and "
                    + "Tools/import_gear_sonic_g1_plant.py first")
        }

        var references: [String: String] = [:]
        let referencesURL = URL(
            fileURLWithPath: bundleDirectory, isDirectory: true)
            .appendingPathComponent("references", isDirectory: true)
        if let directories = try? FileManager.default.contentsOfDirectory(
            at: referencesURL, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]) {
            for directory in directories.sorted(by: {
                $0.lastPathComponent < $1.lastPathComponent
            }) where Self.isGEARSonicReference(at: directory.path) {
                references[directory.lastPathComponent] = directory.path
            }
        }
        if let exactReference = environment["AVBD_GEAR_SONIC_REFERENCE"],
           Self.isGEARSonicReference(at: exactReference) {
            let name = URL(
                fileURLWithPath: exactReference,
                isDirectory: true).lastPathComponent
            references[name] = exactReference
        }
        gearSonicReferenceDirectories = references
        gearSonicReferenceNames = references.keys.sorted()
        guard !gearSonicReferenceNames.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC bundle has no reference clips; install official "
                    + "clip directories under " + bundleDirectory
                    + "/references")
        }

        let requestedClip = environment["AVBD_GEAR_SONIC_CLIP"]
        let persistedClip = UserDefaults.standard.string(
            forKey: "AVBDPolicyReplayGEARSonicReference")
        let preferredNames = [
            requestedClip,
            selectedGEARSonicReference.isEmpty
                ? nil : selectedGEARSonicReference,
            persistedClip,
            "walking_quip_360_R_002__A428",
            "dance_in_da_party_001__A464",
            gearSonicReferenceNames.first,
        ].compactMap { $0 }
        guard let selected = preferredNames.first(where: {
            references[$0] != nil
        }), let referenceDirectory = references[selected] else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC could not select an installed reference clip")
        }

        let session = try GEARSonicG1Session(
            bundleDirectory: bundleDirectory,
            referenceDirectory: referenceDirectory,
            includeVisuals: true)
        gearSonicG1 = session
        gearSonicBundleDirectory = bundleDirectory
        gearSonicReferenceDirectory = referenceDirectory
        selectedGEARSonicReference = selected
        UserDefaults.standard.set(
            selected, forKey: "AVBDPolicyReplayGEARSonicReference")
        let initialState = session.environment.states()[0]
        gearSonicInitialPosition = initialState.root.position
        let first = session.reference.bodyPositions[0]
        let last = session.reference.bodyPositions[
            session.reference.frameCount - 1]
        let referenceDisplacement = F3(
            Float(last[0] - first[0]), Float(last[1] - first[1]),
            Float(last[2] - first[2]))
        gearSonicCourseCenter = gearSonicInitialPosition
            + 0.5 * referenceDisplacement + F3(0, 0, 0.12)
        let planarDistance = simd_length(F3(
            referenceDisplacement.x, referenceDisplacement.y, 0))
        gearSonicCourseDistance = min(
            max(4.5, 2 + 1.5 * planarDistance), 30)
        controlSteps = 0
        episodeFinished = false
        running = true
        let verification = session.policyVerification
        policyStatus = "loaded GEAR-SONIC G1 · MLX/ONNX parity "
            + (verification.passed ? "PASS" : "FAIL")
            + String(format: " (token %.3g, action %.3g)\n",
                     verification.maximumTokenError,
                     verification.maximumActionError)
            + String(session.policy.manifest.weightsSHA256.prefix(12))
            + "… · " + bundleDirectory
        trainingStatus = "static external policy · reference-conditioned "
            + "native MLX inference\nclip " + selected + " · "
            + String(session.reference.frameCount) + " frames at 50 Hz"
    }

    private static func isGEARSonicBundle(at directory: String) -> Bool {
        ["manifest.json", "policy.safetensors", "plant.xml"].allSatisfy {
            FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: directory, isDirectory: true)
                    .appendingPathComponent($0).path)
        }
    }

    private static func isGEARSonicReference(at directory: String) -> Bool {
        ["joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv"]
            .allSatisfy {
                FileManager.default.fileExists(
                    atPath: URL(fileURLWithPath: directory, isDirectory: true)
                        .appendingPathComponent($0).path)
            }
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
        guard arachneRevealController == nil else { return }
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

    func foldArachne() {
        startArachneFold(autoUnfold: false)
    }

    private func startArachneFold(autoUnfold: Bool) {
        guard arachne != nil else { return }
        do {
            // Start from the authored neutral assembly so the reveal is
            // repeatable and never tries to fold an already-fallen robot.
            try restartEpisode()
            // restartEpisode may have replaced a terminal solver and task.
            guard let arachne = self.arachne else {
                throw RLEnvironmentError.invalidConfiguration(
                    "Arachne replay scene is unavailable after reset")
            }
            arachne.environment.setCommissioningTorqueScale(
                Arachne15RevealController.commissioningTorqueScale)
            let measured = arachne.environment.states()[0].jointAngles
            arachneRevealController = Arachne15RevealController(
                initialJointTargets: measured)
            arachneFolded = false
            arachneAutoUnfold = autoUnfold
            interactionStatus =
                "physical fold started · motor targets only · no pose animation"
            episodeFinished = false
            running = true
            accumulator = 0
            lastTime = CACurrentMediaTime()
            refreshStats()
        } catch {
            if self.arachne?.environment.solver.runtimeFailure == nil {
                self.arachne?.environment.setCommissioningTorqueScale(1)
            }
            policyStatus = "reveal failed: \(error.localizedDescription)"
            running = false
        }
    }

    func unfoldArachneAndWalk() {
        guard arachneFolded,
              let reveal = arachneRevealController else { return }
        reveal.beginUnfolding()
        arachneFolded = false
        arachneAutoUnfold = true
        interactionStatus =
            "physical unfold started · learned control resumes after settling"
        running = true
        accumulator = 0
        lastTime = CACurrentMediaTime()
        refreshStats()
    }

    func applyGoalAndReset() {
        do {
            if solver?.runtimeFailure != nil {
                rebuild()
            }
            if let isaacHumanoid, isaacHumanoid.usesPointGoal {
                try installSelectedGoal(in: isaacHumanoid)
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
            if solver?.runtimeFailure != nil {
                rebuild()
            }
            if let isaacHumanoid, isaacHumanoid.usesPointGoal {
                isaacHumanoid.clearGoalOverride(environment: 0)
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
        guard let solver, solver.runtimeFailure == nil else {
            running = false
            interactionStatus = "simulation is stopped after a Metal failure · Reset to rebuild"
            return
        }
        guard supportsBoxThrows else {
            interactionStatus = "this checkpoint scene has no projectile body"
            return
        }
        let launchedSide = nextProjectileSide
        let mass: Float
        if let gearSonicG1 {
            gearSonicG1.environment.throwBoxes(
                environmentIDs: [0], sideSigns: [launchedSide],
                launchDistance: 1.2, speed: Float(boxSpeed))
            mass = gearSonicG1.environment.configuration.projectileMass
        } else if let unitreeH1 {
            unitreeH1.environment.throwBoxes(
                environmentIDs: [0], sideSigns: [launchedSide],
                launchDistance: 1.2, speed: Float(boxSpeed))
            mass = unitreeH1.environment.projectileMass
        } else if let arachne,
                  arachne.environment.hasProjectile(environment: 0) {
            arachne.environment.throwBoxes(
                environmentIDs: [0], sideSigns: [launchedSide],
                launchDistance: 0.35, speed: Float(boxSpeed))
            mass = arachne.environment.projectileMass
        } else if let isaacHumanoid {
            isaacHumanoid.throwRobustnessBoxes(
                environmentIDs: [0], sideSigns: [launchedSide],
                launchDistance: 1.2, speed: Float(boxSpeed))
            mass = isaacHumanoid.environment.projectileMass
        } else {
            return
        }
        let side = launchedSide > 0 ? "left" : "right"
        interactionStatus = String(
            format: "%.1f kg physical box thrown from robot's %@ at %.1f m/s",
            mass, side, boxSpeed)
        nextProjectileSide *= -1
        // A throw while paused should visibly execute instead of leaving the
        // box suspended until the separate Play control is pressed.
        running = true
        accumulator = 0
        lastTime = CACurrentMediaTime()
    }

    /// Polling is intentionally snapshot based. The mutable run root is never
    /// opened while the trainer may be replacing its individual files.
    func pollForLatestCheckpoint(force: Bool = false) {
        refreshTrainingMetrics()
        guard supportsLiveCheckpointLoading else { return }
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
        if let failure = solver?.runtimeFailure {
            // GPUSolver failures are terminal. Rebuild the exact selected task
            // and reload its controller rather than calling reset/setters on
            // poisoned GPU state.
            policyStatus = "rebuilding replay after: \(failure.localizedDescription)"
            rebuild()
            guard let replacement = solver,
                  replacement.runtimeFailure == nil else {
                throw RLEnvironmentError.invalidConfiguration(
                    "replay rebuild failed after terminal Metal error")
            }
            return
        }
        if arachneRevealController != nil {
            arachne?.environment.setCommissioningTorqueScale(1)
        }
        arachneRevealController = nil
        arachneFolded = false
        arachneAutoUnfold = false
        if robot == .gearSonicG1 {
            guard let bundleDirectory = gearSonicBundleDirectory,
                  let referenceDirectory = gearSonicReferenceDirectory else {
                throw RLEnvironmentError.invalidConfiguration(
                    "GEAR-SONIC bundle or reference directory is unavailable")
            }
            let session = try GEARSonicG1Session(
                bundleDirectory: bundleDirectory,
                referenceDirectory: referenceDirectory,
                includeVisuals: true)
            gearSonicG1 = session
            gearSonicInitialPosition = session.environment.states()[0]
                .root.position
            completed = 0; successes = 0
            controlSteps = 0; accumulator = 0
            episodeFinished = false; running = true
            updateCameraTargets()
            applyCameraPreset()
            cameraEpoch += 1
            lastTime = CACurrentMediaTime()
            refreshStats()
            return
        }
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
        if robot.usesClassicalController {
            trainingStatus = "deterministic classical controller · training not required"
            return
        }
        if robot.runtime == .nativeMLX, explicitCheckpointPath != nil {
            if flowReplayController == nil {
                trainingStatus = "static explicit checkpoint replay"
            }
            return
        }
        guard supportsLiveCheckpointLoading else {
            // Installation records the exact source/parity status. Timer-driven
            // native trainer polling must not overwrite it with a generic label.
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
        guard running else { return }
        guard let activeSolver = solver else { return }
        if let failure = activeSolver.runtimeFailure {
            running = false
            policyStatus = "simulation stopped: \(failure.localizedDescription)"
            return
        }
        if !scriptedBoxThrowPerformed,
           let scriptedBoxThrowStep,
           scriptedBoxThrowStep >= 0,
           controlSteps >= scriptedBoxThrowStep,
           supportsBoxThrows {
            scriptedBoxThrowPerformed = true
            throwBox()
        }
        if gearSonicG1 != nil {
            tickGEARSonicG1IfRunning()
            return
        }
        if unitreeH1 != nil {
            tickUnitreeH1IfRunning()
            return
        }
        guard let task, var observation, var result else { return }
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
                if let reveal = arachneRevealController, let arachne {
                    if arachneFolded {
                        try arachne.environment.stepJointTargetsChecked(
                            ContiguousArray(
                                Arachne15RevealController.compactJointTargets),
                            decimation: arachne.configuration.controlDecimation)
                        controlSteps += 1
                        updateCameraTargets()
                        accumulator -= wallStep
                        ticks += 1
                        refreshStats()
                        continue
                    }
                    let targets = reveal.nextJointTargets()
                    try arachne.environment.stepJointTargetsChecked(
                        targets,
                        decimation: arachne.configuration.controlDecimation)
                    controlSteps += 1
                    updateCameraTargets()
                    let compactSettleEnd = reveal.foldingSteps
                        + min(25, reveal.configuration.compactHoldSteps)
                    if !arachneAutoUnfold
                        && reveal.phase == .compactHold
                        && reveal.stepIndex >= compactSettleEnd {
                        arachneFolded = true
                        interactionStatus =
                            "compact pose physically holding · choose Unfold & Walk"
                        accumulator -= wallStep
                        ticks += 1
                        refreshStats()
                        continue
                    }
                    if reveal.isComplete {
                        arachne.environment.setCommissioningTorqueScale(1)
                        try arachne.resumeAfterCommissioning(
                            into: &observation)
                        if let actionProvider {
                            try actionProvider.reset(
                                for: task, observation: observation)
                        }
                        arachneRevealController = nil
                        arachneFolded = false
                        arachneAutoUnfold = false
                        controlSteps = 0
                        interactionStatus = actionProvider == nil
                            ? "unfold complete · deployed pose physically holding"
                            : "unfold complete · locomotion controller has physical state"
                        running = actionProvider != nil
                    }
                    accumulator -= wallStep
                    ticks += 1
                    refreshStats()
                    continue
                }
                let rootVelocityBeforeStep = isaacHumanoid?
                    .environment.states()[0].root.linearVelocity
                    ?? arachne?.environment.states()[0].root.linearVelocity
                guard let actionProvider else {
                    running = false
                    return
                }
                let actions = try actionProvider.actions(
                    for: observation, task: task)
                try task.step(actions: actions, into: &result)
                try actionProvider.resetAfterStep(for: task, result: result)
                let physicalBoxContact =
                    isaacHumanoid?.environment.boxRobotContacts()[0] == true
                    || arachne?.environment.boxRobotContacts()[0] == true
                if physicalBoxContact, let before = rootVelocityBeforeStep {
                    let after = isaacHumanoid?
                        .environment.states()[0].root.linearVelocity
                        ?? arachne?.environment.states()[0].root.linearVelocity
                        ?? before
                    interactionStatus = String(
                        format: "physical contact registered · root Δv %.3f m/s",
                        simd_length(after - before))
                }
                observation = result.observations
                controlSteps += 1
                updateCameraTargets()
                if flowReplayController?.isComplete == true {
                    running = false
                    episodeFinished = true
                    accumulator = 0
                    interactionStatus = switch flowFrontierQualification {
                    case .reusableRecoverySafeFrontier:
                        "recovery-safe reusable frontier reached · Reset or Replay to inspect again"
                    case .measuredFinitePrefix, .none:
                        "finite measured endpoint reached · continuation is not recovery-qualified"
                    }
                    refreshStats()
                    break
                }
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
                if arachneRevealController != nil {
                    if arachne?.environment.solver.runtimeFailure == nil {
                        arachne?.environment.setCommissioningTorqueScale(1)
                    }
                    arachneRevealController = nil
                    arachneFolded = false
                    arachneAutoUnfold = false
                }
                policyStatus = "step failed: \(error.localizedDescription)"
                running = false
                accumulator = 0
                return
            }
            accumulator -= wallStep
            ticks += 1
        }
        self.observation = observation
        self.result = result
        if controlSteps.isMultiple(of: 10) { refreshStats() }
    }

    private func tickGEARSonicG1IfRunning() {
        guard running, let session = gearSonicG1 else { return }
        let now = CACurrentMediaTime()
        let controlPeriod = Double(session.policy.manifest.control.periodSeconds)
        let wallStep = controlPeriod / max(playbackRate, 0.1)
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
                let rootVelocityBeforeStep = session.environment.states()[0]
                    .root.linearVelocity
                _ = try session.step()
                if session.environment.boxRobotContacts()[0] {
                    let after = session.environment.states()[0]
                        .root.linearVelocity
                    interactionStatus = String(
                        format: "physical contact registered · root Δv %.3f m/s",
                        simd_length(after - rootVelocityBeforeStep))
                }
                controlSteps = session.controlSteps
                updateCameraTargets()
                if session.completedReference {
                    running = false
                    episodeFinished = true
                    completed = 1
                    accumulator = 0
                    refreshStats()
                    break
                }
            } catch {
                policyStatus = "step failed: " + error.localizedDescription
                running = false
                accumulator = 0
                return
            }
            accumulator -= wallStep
            ticks += 1
        }
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
                let rootVelocityBeforeStep = session.environment.state()
                    .root.linearVelocity
                _ = try session.step()
                if session.environment.boxRobotContacts()[0] {
                    let after = session.environment.state().root.linearVelocity
                    interactionStatus = String(
                        format: "physical contact registered · root Δv %.3f m/s",
                        simd_length(after - rootVelocityBeforeStep))
                }
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
                accumulator = 0
                return
            }
            accumulator -= wallStep
            ticks += 1
        }
        if controlSteps.isMultiple(of: 10) { refreshStats() }
    }

    func singleStep() {
        guard !episodeFinished, arachneRevealController == nil else { return }
        let wasRunning = running
        running = true
        // Do not fold wall time spent paused into a manual step. Without this
        // reset the 0.1-second catch-up allowance executes three control
        // transitions, making the button visibly skip frames.
        lastTime = CACurrentMediaTime()
        let controlStep: Double
        if let gearSonicG1 {
            controlStep = Double(
                gearSonicG1.policy.manifest.control.periodSeconds)
        } else if robot == .unitreeH1 {
            controlStep = 0.02
        } else {
            controlStep = Double(task?.spec.controlStep ?? 1 / 30)
        }
        accumulator = controlStep / max(playbackRate, 0.1)
        tickIfRunning()
        // A failed or terminal manual transition must remain stopped. In
        // particular, never call the synchronous stats readers on a poisoned
        // solver or restore the pre-step running state after an error.
        guard running, !episodeFinished,
              solver?.runtimeFailure == nil else {
            running = false
            accumulator = 0
            return
        }
        running = wasRunning
        refreshStats()
    }

    private func updateCameraTargets() {
        if let gearSonicG1 {
            let state = gearSonicG1.environment.states()[0]
            replayCameraTarget = state.root.position + F3(0.1, 0, 0.12)
            courseCameraTarget = gearSonicCourseCenter
        } else if let unitreeH1 {
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
        } else if let humanoidBoxCarry {
            let state = humanoidBoxCarry.environment.states()[0]
            let box = humanoidBoxCarry.environment.manipulationStates()[0]
                .object.position
            replayCameraTarget = 0.55 * state.root.position + 0.45 * box
                + F3(0, 0, 0.10)
            courseCameraTarget = 0.5 * (
                box + humanoidBoxCarry.currentPlacementTarget(environment: 0))
                + F3(0, 0, 0.20)
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
        let isBoxCarryCourse = cameraMode == .course
            && humanoidBoxCarry != nil
        solver.settings.cameraDistance = cameraMode == .follow
            ? (isArachne ? 0.65 : (gearSonicG1 != nil ? 2.8 : 3.2))
            : (isArachne ? 1.5
                : (gearSonicG1 != nil ? gearSonicCourseDistance
                    : (isBoxCarryCourse ? 4.2 : 15)))
        solver.settings.cameraTargetX = selectedTarget.x
        solver.settings.cameraTargetY = selectedTarget.y
        solver.settings.cameraTargetZ = selectedTarget.z
        // The carry route turns from +X toward the H1's left (+Y). Looking
        // exactly along Y makes the two tables overlap and falsely reads as a
        // destination behind the source. An oblique course view exposes both
        // axes while preserving the close side view used to judge contacts.
        solver.settings.cameraAzimuth = isBoxCarryCourse
            ? -3 * .pi / 4 : -.pi / 2
        solver.settings.cameraElevation = isBoxCarryCourse
            ? 0.22 : (cameraMode == .follow ? 0.12 : 0.16)
    }

    private func refreshStats() {
        if let gearSonicG1 {
            let state = gearSonicG1.environment.states()[0]
            let report = gearSonicG1.report()
            let actual = state.root.position - gearSonicInitialPosition
            let referenceIndex = max(0, min(
                gearSonicG1.controlSteps - 1,
                gearSonicG1.reference.frameCount - 1))
            let initialReference = gearSonicG1.reference.bodyPositions[0]
            let currentReference = gearSonicG1.reference.bodyPositions[
                referenceIndex]
            let reference = F3(
                Float(currentReference[0] - initialReference[0]),
                Float(currentReference[1] - initialReference[1]),
                Float(currentReference[2] - initialReference[2]))
            let upright = state.root.rotation.act(F3(0, 0, 1)).z
            let totalTime = Float(gearSonicG1.reference.frameCount)
                * gearSonicG1.policy.manifest.control.periodSeconds
            let outcome: String
            if let fall = report.firstFallStep {
                outcome = "FALL RECORDED @ " + String(fall)
            } else if gearSonicG1.completedReference {
                outcome = "FULL CLIP COMPLETE"
            } else {
                outcome = "tracking"
            }
            statsText = String(
                format: "clip %@\n"
                    + "root Δ actual (%+.3f, %+.3f) / reference (%+.3f, %+.3f) m\n"
                    + "joint error mean %.4f rad   max %.4f rad\n"
                    + "14-link error mean %.4f m   max %.4f m\n"
                    + "source eval height %.4f/%.2f m   rotation %.4f/%.2f rad   %@\n"
                    + "height %.3f / min %.3f m   upright %.3f / min %.3f\n"
                    + "frame %d/%d   time %.2f/%.2f s   %@",
                gearSonicReferenceDisplayName(gearSonicG1.referenceName),
                actual.x, actual.y, reference.x, reference.y,
                report.meanJointTrackingErrorRadians,
                report.maximumJointTrackingErrorRadians,
                report.meanRootRelativeBodyTrackingErrorMeters,
                report.maximumRootRelativeBodyTrackingErrorMeters,
                report.maximumSourceCriterionHeightErrorMeters,
                report.sourceHeightFailureThresholdMeters,
                report.maximumRootOrientationErrorRadians,
                report.sourceOrientationFailureThresholdRadians,
                report.sourceCriteriaPassed ? "PASS" : "FAIL",
                state.root.position.z, report.minimumRootHeightMeters,
                upright, report.minimumUprightAlignment,
                gearSonicG1.controlSteps, gearSonicG1.reference.frameCount,
                gearSonicG1.elapsedTime, totalTime, outcome)
        } else if let unitreeH1 {
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
            if let reveal = arachneRevealController {
                let title = arachneFolded
                    ? "ARACHNE COMPACT · HOLDING"
                    : "ARACHNE TRANSFORMATION · \(reveal.phase.rawValue)"
                statsText = String(
                    format: "%@ · %.0f%%\n"
                        + "all motion from 16 torque-limited joints + contact\n"
                        + "height %.3f m   upright %.3f   tick %d/%d",
                    title, reveal.progress * 100,
                    state.root.position.z, up,
                    reveal.stepIndex, reveal.totalSteps)
                return
            }
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
        } else if let humanoidBoxCarry {
            let state = humanoidBoxCarry.environment.states()[0]
            let box = humanoidBoxCarry.environment.manipulationStates()[0]
                .object
            let left = result?.metrics["state/left_hand_contact"]?[0] ?? 0
            let right = result?.metrics["state/right_hand_contact"]?[0] ?? 0
            let clearance = result?.metrics["state/box_clearance_m"]?[0] ?? 0
            let carry = result?.metrics["state/carry_distance_m"]?[0] ?? 0
            let pedestal = result?.metrics[
                "state/box_pedestal_contact"]?[0] ?? 1
            let destination = result?.metrics[
                "state/box_destination_contact"]?[0] ?? 0
            let placement = result?.metrics[
                "state/placement_distance_m"]?[0] ?? .nan
            let released = result?.metrics["state/released"]?[0] ?? 0
            let stableUnsupported = result?.metrics[
                "state/stable_unsupported_steps"]?[0] ?? 0
            let handoff = result?.metrics[
                "state/carry_handoff_progress"]?[0] ?? 0
            let command = result?.metrics[
                "state/carry_command_progress"]?[0] ?? 0
            let outcome: String
            if flowReplayController?.isComplete == true {
                outcome = flowFrontierQualification?.endpointLabel
                    ?? "MEASURED EXPERIMENT ENDPOINT"
            } else if !episodeFinished {
                outcome = "running"
            } else if successes > 0 {
                outcome = "SUCCESS"
            } else if result?.truncated[0] == true {
                outcome = destination > 0.5
                    ? "DESTINATION REACHED · RELEASE INCOMPLETE"
                    : "HORIZON · INCOMPLETE"
            } else {
                outcome = "TERMINATED / FAILURE"
            }
            statsText = String(
                format: "box clearance %+.1f mm   carry %.3f m   goal error %.3f m\n"
                    + "support %@   physical hands %.0f/%.0f   released %@\n"
                    + "unsupported hold %.0f frames   handoff %.0f%%   walk command %.0f%%\n"
                    + "root xy (%+.3f, %+.3f) m   box xy (%+.3f, %+.3f) m   z %.3f m\n"
                    + "phase %@\nframe %d/%d   %@",
                clearance * 1_000, carry, placement,
                destination > 0.5 ? "DESTINATION"
                    : (pedestal > 0.5 ? "SOURCE" : "AIR"),
                left, right, released > 0.5 ? "YES" : "NO",
                stableUnsupported, handoff * 100, command * 100,
                state.root.position.x, state.root.position.y,
                box.position.x, box.position.y, box.position.z,
                flowReplayController?.phaseDescription ?? "policy",
                controlSteps, humanoidBoxCarry.spec.maxEpisodeSteps, outcome)
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
        }
    }
}

struct PolicyReplayLabView: View {
    @StateObject private var model = PolicyReplayModel()
    private let checkpointTimer = Timer.publish(
        every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        HSplitView {
            ZStack {
                PolicyReplayMetalView(model: model)
                if !model.hasReplayScene {
                    Color(nsColor: .windowBackgroundColor)
                    VStack(spacing: 10) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 32))
                            .foregroundStyle(.secondary)
                        Text("Replay unavailable").font(.headline)
                        Text(model.policyStatus.isEmpty
                            ? "No physical replay scene is loaded."
                            : model.policyStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: 440)
                    }
                    .padding(24)
                }
            }
            .frame(minWidth: 650)
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Policy Replay").font(.title2).bold()
                    Text(model.scenarioSummary)
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Picker("Task", selection: $model.robot) {
                        ForEach(PolicyReplayModel.Robot.allCases, id: \.self) {
                            Text($0.displayName).tag($0)
                        }
                    }
                    .pickerStyle(.menu)
                    if model.supportsGEARSonicReferenceSelection {
                        Picker("Reference clip", selection: Binding(
                            get: { model.selectedGEARSonicReference },
                            set: { model.selectGEARSonicReference($0) }
                        )) {
                            if model.gearSonicReferenceNames.isEmpty {
                                Text("No installed clips").tag("")
                            } else {
                                ForEach(model.gearSonicReferenceNames, id: \.self) {
                                    Text(model.gearSonicReferenceDisplayName($0))
                                        .tag($0)
                                }
                            }
                        }
                        .pickerStyle(.menu)
                        .disabled(model.gearSonicReferenceNames.isEmpty)
                        Text("Each entry is a complete official reference sequence. Replay runs to that clip's real final frame unless paused.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack {
                        Button(model.episodeFinished ? "Replay" : (model.running ? "Pause" : "Play")) {
                            model.togglePlayback()
                        }
                        .disabled(model.hasArachneTransformation)
                        Button("Step") { model.singleStep() }
                            .disabled(model.episodeFinished
                                || model.hasArachneTransformation)
                        Button("Reset") { model.resetEpisode() }
                        if model.supportsLiveCheckpointLoading {
                            Button("Load Latest") {
                                model.pollForLatestCheckpoint(force: true)
                            }
                        }
                    }
                    if model.supportsLiveCheckpointLoading {
                        Toggle("Auto-load complete checkpoints",
                               isOn: $model.autoLoadLatest)
                    }
                    Picker("Camera", selection: $model.cameraMode) {
                        ForEach(PolicyReplayModel.CameraMode.allCases, id: \.self) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                    .pickerStyle(.segmented)
                    HStack {
                        Text("Playback").font(.caption)
                        Slider(value: $model.playbackRate, in: 0.1...1)
                        Text(String(format: "%.2fx", model.playbackRate))
                            .font(.caption.monospacedDigit()).frame(width: 42)
                    }
                    Text(model.statsText)
                        .font(.system(.caption, design: .monospaced))
                        .fixedSize(horizontal: false, vertical: true)
                    if model.supportsArachneReveal {
                        Divider()
                        Text("Physical transformation").font(.headline)
                        HStack {
                            Button(model.isArachneTransforming
                                ? "Transforming…" : "Fold") {
                                model.foldArachne()
                            }
                            .disabled(model.hasArachneTransformation)
                            Button("Unfold & Walk") {
                                model.unfoldArachneAndWalk()
                            }
                            .disabled(!model.arachneFolded)
                        }
                        Text("Fold stops and physically holds the compact guard pose. Unfold & Walk deploys through a shared-load crouch and two four-leg waves, restores the walking torque budget, then hands the measured state to the selected controller. During either transformation, no root or link pose is edited.")
                            .font(.caption2).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
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
                    }
                    Divider()
                    Text("Impact test").font(.headline)
                    Text(model.impactModelSummary)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    HStack {
                        Text("Speed").font(.caption)
                            .frame(width: 50, alignment: .leading)
                        Slider(value: $model.boxSpeed,
                               in: model.boxSpeedRange,
                               step: model.boxSpeedStep)
                        Text(String(format: "%.1f", model.boxSpeed))
                            .font(.caption.monospacedDigit()).frame(width: 42)
                    }
                    Button("Throw Physical Box") { model.throwBox() }
                        .disabled(!model.supportsBoxThrows || model.episodeFinished)
                    if !model.supportsBoxThrows {
                        Text("This replay plant does not yet expose a reusable physical projectile.")
                            .font(.caption2).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if !model.interactionStatus.isEmpty {
                        Text(model.interactionStatus)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
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
                    Text(model.replayControlExplanation)
                        .font(.caption2).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(14)
            }
            .frame(minWidth: 520, idealWidth: 600, maxWidth: 720)
        }
        .frame(minWidth: 1200)
        .onReceive(checkpointTimer) { _ in
            model.pollForLatestCheckpoint()
        }
    }
}

private struct PolicyReplayMetalView: NSViewRepresentable {
    @ObservedObject var model: PolicyReplayModel
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> PolicyOrbitMTKView {
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()
        let view = PolicyOrbitMTKView(frame: .zero, device: device)
        if ProcessInfo.processInfo.environment["AVBD_SHOT"] != nil
            || ProcessInfo.processInfo.environment["AVBD_VIDEO_DIR"] != nil {
            view.framebufferOnly = false
        }
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = Renderer.sampleCount
        view.preferredFramesPerSecond = 30
        guard let device else {
            model.reportRenderFailure("no Metal device is available")
            view.isPaused = true
            return view
        }
        do {
            let renderer = try Renderer(device: device, model: model)
            context.coordinator.renderer = renderer
            view.renderer = renderer
            view.delegate = renderer
        } catch {
            model.reportRenderFailure(error.localizedDescription)
            view.isPaused = true
        }
        return view
    }
    func updateNSView(_ view: PolicyOrbitMTKView, context: Context) {
        guard let renderer = context.coordinator.renderer else { return }
        if context.coordinator.cameraMode != model.cameraMode {
            context.coordinator.cameraMode = model.cameraMode
            renderer.azimuth = model.solver?.settings.cameraAzimuth ?? -.pi / 2
            renderer.elevation = model.solver?.settings.cameraElevation
                ?? (model.cameraMode == .follow ? 0.12 : 0.16)
            let isArachne = model.robot == .arachne
                || model.robot == .arachneGoal
                || model.robot == .arachneClassical
            renderer.distance = model.solver?.settings.cameraDistance
                ?? (model.cameraMode == .follow
                    ? (isArachne ? 0.65 : 3.2)
                    : (isArachne ? 1.5 : 15))
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
