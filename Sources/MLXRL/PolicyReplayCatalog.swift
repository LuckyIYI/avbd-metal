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

/// Runtime trust is deliberately separate from catalog qualification. A
/// catalog entry describes the release that may be advertised, while this
/// value describes the bytes selected for this process. Operator overrides,
/// mutable trainer output, and repository-relative development files remain
/// runnable without inheriting a packaged release's qualification.
public enum PolicyReplayRuntimeTrust: String, Sendable, Codable, Equatable {
    case qualifiedPackagedNative
    case verifiedPackagedExternal
    case unverifiedOperatorOverride
    case developmentCheckpoint
    case nonNeuralBaseline
    case unavailable

    public var displayLabel: String {
        switch self {
        case .qualifiedPackagedNative:
            return "QUALIFIED PACKAGED POLICY"
        case .verifiedPackagedExternal:
            return "VERIFIED PACKAGED EXTERNAL POLICY"
        case .unverifiedOperatorOverride:
            return "UNVERIFIED OPERATOR OVERRIDE"
        case .developmentCheckpoint:
            return "UNVERIFIED DEVELOPMENT CHECKPOINT"
        case .nonNeuralBaseline:
            return "NON-NEURAL BASELINE"
        case .unavailable:
            return "POLICY UNAVAILABLE"
        }
    }

    public var isTrustedPackagedRelease: Bool {
        self == .qualifiedPackagedNative
            || self == .verifiedPackagedExternal
    }
}

public enum PolicyReplayCheckpointOrigin: String, Sendable, Codable,
    Equatable
{
    case explicitOverride
    case liveRun
    case applicationBundle
    case repository
}

public struct PolicyReplayCheckpointSource: Sendable, Equatable {
    public var directory: String
    public var origin: PolicyReplayCheckpointOrigin
    public var trust: PolicyReplayRuntimeTrust

    public init(
        directory: String, origin: PolicyReplayCheckpointOrigin,
        trust: PolicyReplayRuntimeTrust
    ) {
        self.directory = directory
        self.origin = origin
        self.trust = trust
    }
}

/// Independent code/catalog trust anchor for the one supported Unitree H1
/// release. The manifest digest authenticates its exact bytes; the individual
/// identities make the commissioned source, converted weights, license, and
/// golden parity sequence explicit and reviewable at the call site.
public struct UnitreeH1ReleaseIdentity: Sendable, Equatable {
    public var manifestSHA256: String
    public var sourceRevision: String
    public var sourceCheckpointSHA256: String
    public var weightsSHA256: String
    public var licenseSHA256: String
    public var goldenSequenceSHA256: String

    public init(
        manifestSHA256: String, sourceRevision: String,
        sourceCheckpointSHA256: String, weightsSHA256: String,
        licenseSHA256: String, goldenSequenceSHA256: String
    ) {
        self.manifestSHA256 = manifestSHA256
        self.sourceRevision = sourceRevision
        self.sourceCheckpointSHA256 = sourceCheckpointSHA256
        self.weightsSHA256 = weightsSHA256
        self.licenseSHA256 = licenseSHA256
        self.goldenSequenceSHA256 = goldenSequenceSHA256
    }
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
    /// Code-pinned release identity. These values must not be derived from the
    /// checkpoint currently being opened, or a self-consistent replacement
    /// could certify itself.
    public var expectedTaskRevision: Int?
    public var expectedCheckpointFingerprint: String?
    /// Digest of the exact deployment-manifest bytes, including formatting.
    public var expectedDeploymentManifestSHA256: String?
    public var unitreeH1ReleaseIdentity: UnitreeH1ReleaseIdentity?

    public init(
        selectionID: String, displayName: String, taskID: String,
        runtime: PolicyReplayRuntime,
        checkpointRelativeDirectory: String?,
        qualification: PolicyReplayQualification,
        evidenceRelativePath: String?,
        acceptanceAggregateRelativePath: String? = nil,
        deploymentManifestRelativePath: String? = nil,
        expectedTaskRevision: Int? = nil,
        expectedCheckpointFingerprint: String? = nil,
        expectedDeploymentManifestSHA256: String? = nil,
        unitreeH1ReleaseIdentity: UnitreeH1ReleaseIdentity? = nil
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
        self.expectedTaskRevision = expectedTaskRevision
        self.expectedCheckpointFingerprint = expectedCheckpointFingerprint
        self.expectedDeploymentManifestSHA256 =
            expectedDeploymentManifestSHA256
        self.unitreeH1ReleaseIdentity = unitreeH1ReleaseIdentity
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

    /// Resolve provenance and trust together. An explicit source remains
    /// authoritative, but it is never allowed to borrow a catalog release's
    /// qualification. Only an app-bundle source can be advertised as the
    /// accepted/verified release; live and repository sources are development
    /// inputs even when their bytes happen to match today.
    public static func sources(
        explicit: String?, live: String?, bundled: String?,
        repository: String?, entry: PolicyReplayCatalogEntry
    ) -> [PolicyReplayCheckpointSource] {
        if let explicit {
            return [.init(
                directory: explicit, origin: .explicitOverride,
                trust: .unverifiedOperatorOverride)]
        }
        var result = [PolicyReplayCheckpointSource]()
        if let live {
            result.append(.init(
                directory: live, origin: .liveRun,
                trust: .developmentCheckpoint))
        }
        if let bundled {
            let trust: PolicyReplayRuntimeTrust = switch entry.qualification {
            case .accepted: .qualifiedPackagedNative
            case .externalParityVerified: .verifiedPackagedExternal
            case .development, .requalificationRequired:
                .developmentCheckpoint
            case .nonNeuralBaseline: .nonNeuralBaseline
            }
            result.append(.init(
                directory: bundled, origin: .applicationBundle,
                trust: trust))
        }
        if let repository {
            result.append(.init(
                directory: repository, origin: .repository,
                trust: .developmentCheckpoint))
        }
        return result
    }
}

/// Single source of truth for the polished replay surface. Experimental tasks
/// remain trainable through the generic registry without appearing here.
public enum PolicyReplayCatalog {
    public static let entries: [PolicyReplayCatalogEntry] = {
        let entries: [PolicyReplayCatalogEntry] = [
            .init(
                selectionID: "humanoid-isaac-flat-v2",
                displayName: "H1 Flat",
                taskID: "humanoid-isaac-flat-v0",
                runtime: .nativeMLX,
                checkpointRelativeDirectory: "humanoid-isaac-flat-v2",
                qualification: .accepted,
                evidenceRelativePath:
                    "checkpoints/humanoid-isaac-flat-v2/requalification-manifest.json",
                acceptanceAggregateRelativePath:
                    "checkpoints/humanoid-isaac-flat-v2/qualification/aggregate.json",
                deploymentManifestRelativePath:
                    "checkpoints/humanoid-isaac-flat-v2/deployment-manifest.json",
                expectedTaskRevision: 2_000_011,
                expectedCheckpointFingerprint:
                    "00bc782d1845ddde94282b46f0d7fa2732feeb4a8e52215a5abe62128bccc756",
                expectedDeploymentManifestSHA256:
                    "cb04233bd11bcc8dc3e0d2e1f0d6cc2e1ec27d4318344c7ea021d8b117be5d59"),
            .init(
                selectionID: "unitree-h1-sim2sim-v0",
                displayName: "Unitree H1 Sim2Sim",
                taskID: "unitree-h1-sim2sim-v0",
                runtime: .unitreeRecurrentMLX,
                checkpointRelativeDirectory: "external/unitree-h1",
                qualification: .externalParityVerified,
                evidenceRelativePath: "checkpoints/external/unitree-h1/manifest.json",
                unitreeH1ReleaseIdentity: .init(
                    manifestSHA256:
                        "9f434828cf2b2ede587bced686a22d30c3df6b048e631e94641bafeb7a45d117",
                    sourceRevision:
                        "276801e46c5d433564f24658bac64f254b7d2d4b",
                    sourceCheckpointSHA256:
                        "44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb",
                    weightsSHA256:
                        "cb51db3e4ccbecc0d9a863173640f8cb8b5a5fb821bc1db9024c7957297ff4ee",
                    licenseSHA256:
                        "98335465f43a20b5850e4651db6e74c4aa1e9fc8e8813d38f345178045c0da50",
                    goldenSequenceSHA256:
                        "705e5d5bad46f696b4617ea440cfda1b4ff51b9504d949c5384e5d73cc14015b")),
            .init(
                selectionID: "arachne15-velocity-v1",
                displayName: "Arachne Straight Walk",
                taskID: "arachne15-velocity-v0",
                runtime: .nativeMLX,
                checkpointRelativeDirectory: "arachne15-velocity-v1",
                qualification: .accepted,
                evidenceRelativePath:
                    "checkpoints/arachne15-velocity-v1/requalification-manifest.json",
                acceptanceAggregateRelativePath:
                    "checkpoints/arachne15-velocity-v1/qualification/nominal/aggregate.json",
                deploymentManifestRelativePath:
                    "checkpoints/arachne15-velocity-v1/deployment-manifest.json",
                expectedTaskRevision: 2_000_006,
                expectedCheckpointFingerprint:
                    "97f79641c8b7acf87c903b9d6baf739a5dc3c2536e52cb0e44121260133d79d5",
                expectedDeploymentManifestSHA256:
                    "7295cf74dc9576a8b2bce74eeae0beb2932af96c5ff0617b0b99b82796b5039c"),
            .init(
                selectionID: "arachne15-goal-v1",
                displayName: "Arachne Goal",
                taskID: "arachne15-goal-v0",
                runtime: .nativeMLX,
                checkpointRelativeDirectory: "arachne15-goal-v1",
                qualification: .accepted,
                evidenceRelativePath:
                    "checkpoints/arachne15-goal-v1/requalification-manifest.json",
                acceptanceAggregateRelativePath:
                    "checkpoints/arachne15-goal-v1/qualification/nominal/aggregate.json",
                deploymentManifestRelativePath:
                    "checkpoints/arachne15-goal-v1/deployment-manifest.json",
                expectedTaskRevision: 2_000_006,
                expectedCheckpointFingerprint:
                    "923e07c286f4fdb186b30a6fd95469e6848f4fec4ca1e3811320424b94c9dc02",
                expectedDeploymentManifestSHA256:
                    "e7d747a41b3f724940bbe42d92dc38de8798dafbc7d39909e4ad1cf10ae1e127"),
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
                        && $0.deploymentManifestRelativePath != nil
                        && $0.expectedTaskRevision != nil
                        && $0.expectedCheckpointFingerprint != nil
                        && $0.expectedDeploymentManifestSHA256 != nil)
            }, "accepted replay entries require pinned deployment identity")
        precondition(
            entries.allSatisfy {
                $0.qualification != .externalParityVerified
                    || ($0.runtime == .unitreeRecurrentMLX
                        && $0.unitreeH1ReleaseIdentity != nil)
            }, "verified external entries require a pinned release identity")
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
