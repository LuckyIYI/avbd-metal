/// Runtime used by one curated Policy Replay entry. Learned entries must name
/// a packaged checkpoint directory; the classical baseline must not.
public enum PolicyReplayRuntime: String, Sendable, Codable {
    case nativeMLX
    case unitreeRecurrentMLX
    /// Static external policy whose action is conditioned on a reference clip.
    /// Its locally imported weights are not part of the native PPO checkpoint
    /// discovery or hot-reload path.
    case externalReferenceMLX
    case classicalController
}

public enum PolicyReplayQualification: String, Sendable, Codable {
    case externalParityVerified
    case accepted
    case development
    /// The immutable packaged checkpoint predates the current task physics
    /// revision and is retained only as a historical migration boundary.
    case requalificationRequired
    case nonNeuralBaseline
}

public struct PolicyReplayCatalogEntry: Sendable, Equatable {
    public var selectionID: String
    public var displayName: String
    public var taskID: String
    public var runtime: PolicyReplayRuntime
    /// Relative to the app's `checkpoints` resource directory.
    public var checkpointRelativeDirectory: String?
    public var qualification: PolicyReplayQualification
    /// Repository-relative parity/evaluation evidence or deterministic,
    /// source-locked importer for a locally installed external artifact.
    public var evidenceRelativePath: String?
    /// A decoded, internally consistent robustness aggregate is required for
    /// every entry advertised as accepted.
    public var acceptanceAggregateRelativePath: String?
    /// Accepted native policies must also name the immutable deployment
    /// manifest that binds their runtime bytes to the evaluated fingerprint.
    public var deploymentManifestRelativePath: String?

    public init(
        selectionID: String, displayName: String, taskID: String,
        runtime: PolicyReplayRuntime,
        checkpointRelativeDirectory: String?,
        qualification: PolicyReplayQualification,
        evidenceRelativePath: String?,
        acceptanceAggregateRelativePath: String? = nil,
        deploymentManifestRelativePath: String? = nil
    ) {
        self.selectionID = selectionID
        self.displayName = displayName
        self.taskID = taskID
        self.runtime = runtime
        self.checkpointRelativeDirectory = checkpointRelativeDirectory
        self.qualification = qualification
        self.evidenceRelativePath = evidenceRelativePath
        self.acceptanceAggregateRelativePath =
            acceptanceAggregateRelativePath
        self.deploymentManifestRelativePath = deploymentManifestRelativePath
    }
}

/// Shared source-precedence rule for replay checkpoints. An explicit path is
/// an auditable provenance boundary: an invalid override must fail rather than
/// silently executing different live, bundled, or repository weights.
public enum PolicyReplayCheckpointResolution {
    public static func candidates(
        explicit: String?, fallbacks: [String?]
    ) -> [String] {
        if let explicit { return [explicit] }
        return fallbacks.compactMap { $0 }
    }
}

/// Single source of truth for the polished replay surface. Experimental tasks
/// remain trainable through the generic registry without appearing here.
public enum PolicyReplayCatalog {
    public static let entries: [PolicyReplayCatalogEntry] = {
        let entries: [PolicyReplayCatalogEntry] = [
            .init(
                selectionID: "unitree-h1-sim2sim-v0",
                displayName: "Unitree H1 Sim2Sim",
                taskID: "unitree-h1-sim2sim-v0",
                runtime: .unitreeRecurrentMLX,
                checkpointRelativeDirectory: "external/unitree-h1",
                qualification: .externalParityVerified,
                evidenceRelativePath: "checkpoints/external/unitree-h1/manifest.json"),
            .init(
                selectionID: "gear-sonic-g1-reference-v0",
                displayName: "GEAR-SONIC G1",
                taskID: "gear-sonic-g1-reference-v0",
                runtime: .externalReferenceMLX,
                checkpointRelativeDirectory: "external/gear-sonic-g1",
                qualification: .development,
                evidenceRelativePath: "Tools/import_gear_sonic_policy.py"),
            .init(
                selectionID: "arachne15-classical-goal-v0",
                displayName: "Arachne Classical",
                taskID: "arachne15-goal-v0",
                runtime: .classicalController,
                checkpointRelativeDirectory: nil,
                qualification: .nonNeuralBaseline,
                evidenceRelativePath: nil),
        ]
        precondition(
            entries.allSatisfy {
                $0.qualification != .accepted
                    || ($0.runtime == .nativeMLX
                        && $0.acceptanceAggregateRelativePath != nil
                        && $0.deploymentManifestRelativePath != nil)
            }, "accepted replay entries require aggregate and deployment evidence")
        return entries
    }()

    /// Immutable tracked checkpoints retained for provenance and migration.
    /// These entries are deliberately excluded from `entries`, so they cannot
    /// appear in a replay picker or be selected by `entry(selectionID:)`.
    public static let historicalEntries: [PolicyReplayCatalogEntry] = [
        .init(
            selectionID: "humanoid-isaac-flat-v1",
            displayName: "H1 Flat (Epoch 1)",
            taskID: "humanoid-isaac-flat-v0",
            runtime: .nativeMLX,
            checkpointRelativeDirectory: "humanoid-isaac-flat-v1",
            qualification: .requalificationRequired,
            evidenceRelativePath:
                "checkpoints/humanoid-isaac-flat-v1/requalification-manifest.json",
            acceptanceAggregateRelativePath:
                "checkpoints/humanoid-isaac-flat-v1/qualification/aggregate.json",
            deploymentManifestRelativePath:
                "checkpoints/humanoid-isaac-flat-v1/deployment-manifest.json"),
        .init(
            selectionID: "humanoid-isaac-flat-v0",
            displayName: "H1 Flat (Legacy Hulls)",
            taskID: "humanoid-isaac-flat-v0",
            runtime: .nativeMLX,
            checkpointRelativeDirectory: "humanoid-isaac-flat-v0",
            qualification: .requalificationRequired,
            evidenceRelativePath:
                "checkpoints/humanoid-isaac-flat-v0/evaluation.json"),
        .init(
            selectionID: "humanoid-isaac-goal-v0",
            displayName: "H1 Goal (Epoch 1)",
            taskID: "humanoid-isaac-goal-v0",
            runtime: .nativeMLX,
            checkpointRelativeDirectory: "humanoid-isaac-goal-v0",
            qualification: .requalificationRequired,
            evidenceRelativePath:
                "checkpoints/humanoid-isaac-goal-v0/evaluation.json"),
        .init(
            selectionID: "arachne15-velocity-v0",
            displayName: "Arachne Straight Walk (Epoch 1)",
            taskID: "arachne15-velocity-v0",
            runtime: .nativeMLX,
            checkpointRelativeDirectory: "arachne15-velocity-v0",
            qualification: .requalificationRequired,
            evidenceRelativePath:
                "checkpoints/arachne15-velocity-v0/evaluation.json"),
        .init(
            selectionID: "arachne15-goal-v0",
            displayName: "Arachne Goal (Epoch 1)",
            taskID: "arachne15-goal-v0",
            runtime: .nativeMLX,
            checkpointRelativeDirectory: "arachne15-goal-v0",
            qualification: .requalificationRequired,
            evidenceRelativePath: "Robots/Arachne15/qualification/"
                + "arachne15-goal-r6-update-000020/"
                + "aggregate-update-000020.json",
            acceptanceAggregateRelativePath:
                "Robots/Arachne15/qualification/"
                + "arachne15-goal-r6-update-000020/"
                + "aggregate-update-000020.json",
            deploymentManifestRelativePath:
                "checkpoints/arachne15-goal-v0/deployment-manifest.json"),
    ]

    /// Complete repository inventory, including nonselectable historical
    /// lineage. This is broader than the app's runtime package allowlist.
    public static var allDeclaredEntries: [PolicyReplayCatalogEntry] {
        entries + historicalEntries
    }

    public static func entry(selectionID: String)
        -> PolicyReplayCatalogEntry?
    {
        entries.first { $0.selectionID == selectionID }
    }

    public static func historicalEntry(selectionID: String)
        -> PolicyReplayCatalogEntry?
    {
        historicalEntries.first { $0.selectionID == selectionID }
    }

    public static var nativeLearnedEntries: [PolicyReplayCatalogEntry] {
        entries.filter { $0.runtime == .nativeMLX }
    }
}
