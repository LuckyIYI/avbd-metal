import SimCore
import PhysicsAVBD
import Robotics
import RL
import CryptoKit
import Foundation

/// A qualification plan is fixed before a transferred policy is evaluated.
/// Keeping the seed matrix in the candidate prevents replacing an
/// inconvenient reset stream after observing its result.
public struct VectorPolicyRequalificationPlan: Codable, Sendable, Equatable {
    public var evaluationSeeds: [UInt64]
    public var evaluationEnvironments: Int
    public var episodesPerReport: Int

    public init(
        evaluationSeeds: [UInt64], evaluationEnvironments: Int,
        episodesPerReport: Int = 512
    ) {
        self.evaluationSeeds = evaluationSeeds
        self.evaluationEnvironments = evaluationEnvironments
        self.episodesPerReport = episodesPerReport
    }
}

/// One immutable evaluation distribution in a multi-suite qualification.
/// The complete configuration is recorded instead of a patch so omitted
/// options cannot silently inherit different defaults at publication time.
public struct VectorPolicyRequalificationSuitePlan:
    Codable, Sendable, Equatable
{
    public var id: String
    public var evaluationSeeds: [UInt64]
    public var evaluationEnvironments: Int
    public var episodesPerReport: Int
    public var evaluationTaskConfiguration: [String: Float]
    /// Exact sorted keys that differ from the checkpoint task contract.
    public var changedConfigurationFields: [String]
    public var evaluationCriteria: VectorPolicyRequalificationCriteria

    public init(
        id: String, evaluationSeeds: [UInt64],
        evaluationEnvironments: Int, episodesPerReport: Int = 512,
        evaluationTaskConfiguration: [String: Float],
        changedConfigurationFields: [String],
        evaluationCriteria: RLEvaluationCriteria
    ) {
        self.id = id
        self.evaluationSeeds = evaluationSeeds
        self.evaluationEnvironments = evaluationEnvironments
        self.episodesPerReport = episodesPerReport
        self.evaluationTaskConfiguration = evaluationTaskConfiguration
        self.changedConfigurationFields = changedConfigurationFields
        self.evaluationCriteria = .init(evaluationCriteria)
    }

    fileprivate var legacyPlan: VectorPolicyRequalificationPlan {
        .init(
            evaluationSeeds: evaluationSeeds,
            evaluationEnvironments: evaluationEnvironments,
            episodesPerReport: episodesPerReport)
    }
}

/// A predeclared cross-distribution robustness requirement. A positive drop
/// means the evaluated suite performed worse than the baseline suite.
public struct VectorPolicyRequalificationComparison:
    Codable, Sendable, Equatable
{
    public var baselineSuite: String
    public var evaluatedSuite: String
    public var maximumPooledSuccessRateDrop: Float

    public init(
        baselineSuite: String, evaluatedSuite: String,
        maximumPooledSuccessRateDrop: Float
    ) {
        self.baselineSuite = baselineSuite
        self.evaluatedSuite = evaluatedSuite
        self.maximumPooledSuccessRateDrop = maximumPooledSuccessRateDrop
    }
}

/// Optional schema-v2 extension for evaluating one checkpoint on multiple,
/// explicitly different task distributions.
public struct VectorPolicyRequalificationMatrix:
    Codable, Sendable, Equatable
{
    public var suites: [VectorPolicyRequalificationSuitePlan]
    public var comparisons: [VectorPolicyRequalificationComparison]

    public init(
        suites: [VectorPolicyRequalificationSuitePlan],
        comparisons: [VectorPolicyRequalificationComparison]
    ) {
        self.suites = suites
        self.comparisons = comparisons
    }
}

/// Exact task-owned thresholds fixed before any qualification seed is run.
/// Publication compares the live task contract with this snapshot, preventing
/// post-result threshold shopping through the public API or CLI.
public struct VectorPolicyRequalificationCriteria:
    Codable, Sendable, Equatable
{
    public var minimumSuccessRate: Float
    public var minimumMeanEpisodeLengthFraction: Float
    public var minimumTaskMetrics: [String: Float]
    public var maximumTaskMetrics: [String: Float]

    public init(_ criteria: RLEvaluationCriteria) {
        minimumSuccessRate = criteria.minimumSuccessRate
        minimumMeanEpisodeLengthFraction =
            criteria.minimumMeanEpisodeLengthFraction
        minimumTaskMetrics = criteria.minimumTaskMetrics
        maximumTaskMetrics = criteria.maximumTaskMetrics
    }

}

public struct VectorPolicyRequalificationFileEvidence:
    Codable, Sendable, Equatable
{
    public var file: String
    public var sha256: String
}

public struct VectorPolicyRequalificationReportEvidence:
    Codable, Sendable, Equatable
{
    public var evaluationSeed: UInt64
    public var file: String
    public var sha256: String
}

public struct VectorPolicyRequalificationEvidence:
    Codable, Sendable, Equatable
{
    public var reports: [VectorPolicyRequalificationReportEvidence]
    public var aggregate: VectorPolicyRequalificationFileEvidence
    /// Schema-v1 keeps this absent, preserving historical manifest bytes.
    /// In schema-v2 the primary suite still uses the legacy fields above and
    /// every remaining suite is named here.
    public var additionalSuites: [VectorPolicyRequalificationSuiteEvidence]?

    public init(
        reports: [VectorPolicyRequalificationReportEvidence],
        aggregate: VectorPolicyRequalificationFileEvidence,
        additionalSuites: [VectorPolicyRequalificationSuiteEvidence]? = nil
    ) {
        self.reports = reports
        self.aggregate = aggregate
        self.additionalSuites = additionalSuites
    }
}

public struct VectorPolicyRequalificationSuiteEvidence:
    Codable, Sendable, Equatable
{
    public var id: String
    public var reports: [VectorPolicyRequalificationReportEvidence]
    public var aggregate: VectorPolicyRequalificationFileEvidence

    public init(
        id: String,
        reports: [VectorPolicyRequalificationReportEvidence],
        aggregate: VectorPolicyRequalificationFileEvidence
    ) {
        self.id = id
        self.reports = reports
        self.aggregate = aggregate
    }
}

/// Publication-time file locations for one non-primary suite. Paths are read
/// only after the sealed candidate's matrix has been validated.
public struct VectorPolicyRequalificationSuiteInput: Sendable, Equatable {
    public var id: String
    public var evaluationReportPaths: [String]
    public var aggregatePath: String

    public init(
        id: String, evaluationReportPaths: [String], aggregatePath: String
    ) {
        self.id = id
        self.evaluationReportPaths = evaluationReportPaths
        self.aggregatePath = aggregatePath
    }
}

/// A live registry-constructed task contract supplied independently from the
/// manifest at preparation, publication, and verification boundaries.
public struct VectorPolicyRequalificationSuiteContract:
    Sendable, Equatable
{
    public var id: String
    public var taskSpec: RLTaskSpec
    public var evaluationCriteria: RLEvaluationCriteria

    public init(
        id: String, taskSpec: RLTaskSpec,
        evaluationCriteria: RLEvaluationCriteria
    ) {
        self.id = id
        self.taskSpec = taskSpec
        self.evaluationCriteria = evaluationCriteria
    }
}

/// Machine-readable lineage for a zero-update task-revision transfer.
///
/// `taskRevision` in ordinary checkpoint metadata is the exact runtime
/// contract. The source fields below retain the distinct contract that
/// produced the weights, while the zero training counters make it impossible
/// to mistake requalification for retraining on the target plant.
///
/// This record provides deterministic integrity checks for a trusted operator;
/// it is not a signature or remote attestation that the declared simulations
/// ran on a particular executable or machine.
public struct VectorPolicyRequalificationManifest:
    Codable, Sendable, Equatable
{
    public var schemaVersion: Int
    public var task: String
    public var sourceTaskRevision: Int
    public var targetTaskRevision: Int
    public var parentCheckpointDirectory: String
    public var candidateCheckpointDirectory: String
    public var parentCheckpointFingerprint: String
    public var candidateCheckpointFingerprint: String
    public var parentPolicySHA256: String
    public var candidatePolicySHA256: String
    public var parentMetadataSHA256: String
    public var candidateMetadataSHA256: String
    public var parentTrainingStateSHA256: String
    public var candidateTrainingStateSHA256: String
    /// Commit identity declared by the operator before evaluation. This is
    /// provenance, not a cryptographic attestation of the running executable.
    public var declaredSourceCommit: String
    public var taskConfiguration: [String: Float]
    public var observationDimension: Int
    public var actionDimension: Int
    public var simulationStepSeconds: Float
    public var controlDecimation: Int
    public var maxEpisodeSteps: Int
    public var inferenceBatchSize: Int
    public var parentTrainingUpdates: Int
    public var parentTrainingEnvironmentSteps: Int
    public var targetTrainingUpdates: Int
    public var targetTrainingEnvironmentSteps: Int
    public var changedFields: [String]
    public var qualificationPlan: VectorPolicyRequalificationPlan
    /// Present only for schema-v2 multi-distribution qualifications. The
    /// first suite is mirrored by `qualificationPlan` for schema-v1 readers.
    public var qualificationMatrix: VectorPolicyRequalificationMatrix?
    public var evaluationCriteria: VectorPolicyRequalificationCriteria
    public var qualification: VectorPolicyRequalificationEvidence?
}

/// Creates and seals auditable, optimizer-free policy revision transfers.
/// Preparation and publication both use a sibling staging directory followed
/// by one rename, and neither operation replaces an existing destination.
public enum VectorPolicyRequalification {
    public static let manifestFileName = "requalification-manifest.json"
    private static let expectedChangedFields = [
        "metadata.taskRevision",
        "metadata.inferenceBatchSize",
        "metadata.ppo.initializationCheckpoint",
        "metadata.ppo.updates",
        "training-state.completedUpdates",
        "training-state.environmentSteps",
        "training-state.optimizerSteps",
        "training-state.adaptiveLearningRate",
    ]

    @discardableResult
    public static func prepare(
        targetSpec: RLTaskSpec,
        parentCheckpointDirectory: String,
        outputDirectory: String,
        expectedParentCheckpointFingerprint: String,
        inferenceBatchSize: Int,
        declaredSourceCommit: String,
        qualificationPlan: VectorPolicyRequalificationPlan,
        evaluationCriteria: RLEvaluationCriteria,
        qualificationMatrix: VectorPolicyRequalificationMatrix? = nil,
        suiteContracts: [VectorPolicyRequalificationSuiteContract] = []
    ) throws -> VectorPolicyRequalificationManifest {
        try validate(plan: qualificationPlan)
        let frozenCriteria = VectorPolicyRequalificationCriteria(
            evaluationCriteria)
        try validate(criteria: frozenCriteria)
        try validate(
            matrix: qualificationMatrix, primaryPlan: qualificationPlan,
            primaryCriteria: frozenCriteria, targetSpec: targetSpec,
            suiteContracts: suiteContracts)
        guard inferenceBatchSize == qualificationPlan.evaluationEnvironments else {
            throw invalid(
                "inference batch size must equal the declared evaluation replica count")
        }
        try validateDeclaredSourceCommit(declaredSourceCommit)

        let manager = FileManager.default
        let parent = URL(
            fileURLWithPath: parentCheckpointDirectory, isDirectory: true)
        let destination = URL(
            fileURLWithPath: outputDirectory, isDirectory: true)
        try requireAbsent(destination, manager: manager)
        guard !isSameOrDescendant(destination, of: parent) else {
            throw invalid(
                "requalification output must not equal or be nested in its parent")
        }

        let parentMetadataData = try requiredData(
            parent.appendingPathComponent("metadata.json"))
        let parentPolicyData = try requiredData(
            parent.appendingPathComponent("policy.safetensors"))
        let parentTrainingStateData = try requiredData(
            parent.appendingPathComponent("training-state.json"))
        let parentMetadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self, from: parentMetadataData)
        let parentTrainingState = try JSONDecoder().decode(
            VectorPPOTrainingState.self, from: parentTrainingStateData)
        guard let sourceRevision = parentMetadata.taskRevision else {
            throw invalid("parent checkpoint has no explicit task revision")
        }
        let parentFingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: parent.path)
        guard parentFingerprint == expectedParentCheckpointFingerprint else {
            throw invalid("parent checkpoint fingerprint does not match the expected value")
        }
        let mismatches = parentMetadata.compatibilityMismatches(with: targetSpec)
        guard sourceRevision != targetSpec.revision,
              mismatches.count == 1,
              mismatches[0].hasPrefix("revision ") else {
            throw invalid(
                "zero-update requalification requires revision to be the only "
                    + "checkpoint/task mismatch; found: "
                    + (mismatches.isEmpty ? "none" : mismatches.joined(separator: "; ")))
        }

        let candidateMetadata = expectedCandidateMetadata(
            parent: parentMetadata, targetSpec: targetSpec,
            inferenceBatchSize: inferenceBatchSize,
            parentCheckpointDirectory: parentCheckpointDirectory)
        let candidateTrainingState = zeroUpdateTrainingState()
        let candidateMetadataData = try encoded(candidateMetadata)
        let candidateTrainingStateData = try encoded(candidateTrainingState)

        let destinationParent = destination.deletingLastPathComponent()
        try manager.createDirectory(
            at: destinationParent, withIntermediateDirectories: true)
        let staging = destinationParent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true)
        try manager.createDirectory(
            at: staging, withIntermediateDirectories: false)
        var published = false
        defer {
            if !published { try? manager.removeItem(at: staging) }
        }

        try candidateMetadataData.write(
            to: staging.appendingPathComponent("metadata.json"), options: .atomic)
        try parentPolicyData.write(
            to: staging.appendingPathComponent("policy.safetensors"),
            options: .atomic)
        try candidateTrainingStateData.write(
            to: staging.appendingPathComponent("training-state.json"),
            options: .atomic)
        let copiedPolicy = try requiredData(
            staging.appendingPathComponent("policy.safetensors"))
        guard copiedPolicy == parentPolicyData else {
            throw invalid("candidate policy bytes differ from the parent")
        }
        let candidateFingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: staging.path)
        let parentPolicyHash = sha256(parentPolicyData)
        let candidatePolicyHash = sha256(copiedPolicy)
        guard candidatePolicyHash == parentPolicyHash,
              candidateFingerprint != parentFingerprint else {
            throw invalid("candidate policy SHA-256 differs from the parent")
        }

        let manifest = VectorPolicyRequalificationManifest(
            schemaVersion: qualificationMatrix == nil ? 1 : 2,
            task: targetSpec.id,
            sourceTaskRevision: sourceRevision,
            targetTaskRevision: targetSpec.revision,
            parentCheckpointDirectory: parentCheckpointDirectory,
            candidateCheckpointDirectory: outputDirectory,
            parentCheckpointFingerprint: parentFingerprint,
            candidateCheckpointFingerprint: candidateFingerprint,
            parentPolicySHA256: parentPolicyHash,
            candidatePolicySHA256: candidatePolicyHash,
            parentMetadataSHA256: sha256(parentMetadataData),
            candidateMetadataSHA256: sha256(candidateMetadataData),
            parentTrainingStateSHA256: sha256(parentTrainingStateData),
            candidateTrainingStateSHA256: sha256(candidateTrainingStateData),
            declaredSourceCommit: declaredSourceCommit,
            taskConfiguration: targetSpec.configurationValues,
            observationDimension: targetSpec.observation.elementCount,
            actionDimension: targetSpec.action.elementCount,
            simulationStepSeconds: targetSpec.simulationStep,
            controlDecimation: targetSpec.controlDecimation,
            maxEpisodeSteps: targetSpec.maxEpisodeSteps,
            inferenceBatchSize: inferenceBatchSize,
            parentTrainingUpdates: parentTrainingState.completedUpdates,
            parentTrainingEnvironmentSteps: parentTrainingState.environmentSteps,
            targetTrainingUpdates: 0,
            targetTrainingEnvironmentSteps: 0,
            changedFields: expectedChangedFields,
            qualificationPlan: qualificationPlan,
            qualificationMatrix: qualificationMatrix,
            evaluationCriteria: frozenCriteria,
            qualification: nil)
        try encoded(manifest).write(
            to: staging.appendingPathComponent(manifestFileName),
            options: .atomic)
        try manager.moveItem(at: staging, to: destination)
        published = true
        return manifest
    }

    @discardableResult
    public static func publish(
        targetSpec: RLTaskSpec,
        evaluationCriteria: RLEvaluationCriteria,
        candidateDirectory: String,
        parentCheckpointDirectory: String,
        evaluationReportPaths: [String],
        aggregatePath: String,
        additionalSuiteInputs: [VectorPolicyRequalificationSuiteInput] = [],
        suiteContracts: [VectorPolicyRequalificationSuiteContract] = [],
        outputDirectory: String
    ) throws -> VectorPolicyRequalificationManifest {
        let manager = FileManager.default
        let candidate = URL(
            fileURLWithPath: candidateDirectory, isDirectory: true)
        let parent = URL(
            fileURLWithPath: parentCheckpointDirectory, isDirectory: true)
        let output = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        try requireAbsent(output, manager: manager)
        guard !isSameOrDescendant(output, of: candidate),
              !isSameOrDescendant(output, of: parent) else {
            throw invalid(
                "publication output must not equal or be nested in candidate or parent")
        }

        let preparationData = try requiredData(
            candidate.appendingPathComponent(manifestFileName))
        var manifest = try JSONDecoder().decode(
            VectorPolicyRequalificationManifest.self, from: preparationData)
        guard [1, 2].contains(manifest.schemaVersion),
              manifest.qualification == nil else {
            throw invalid(
                "candidate does not contain an unsealed supported manifest")
        }
        try validate(plan: manifest.qualificationPlan)
        try validate(criteria: manifest.evaluationCriteria)
        try validate(
            matrix: manifest.qualificationMatrix,
            primaryPlan: manifest.qualificationPlan,
            primaryCriteria: manifest.evaluationCriteria,
            targetSpec: targetSpec, suiteContracts: suiteContracts)
        guard manifest.schemaVersion
                == (manifest.qualificationMatrix == nil ? 1 : 2) else {
            throw invalid("requalification schema does not match its matrix")
        }
        try validateManifestContract(manifest, targetSpec: targetSpec)
        guard manifest.evaluationCriteria
                == VectorPolicyRequalificationCriteria(evaluationCriteria) else {
            throw invalid(
                "task-owned evaluation criteria changed after preparation")
        }

        let validated = try validateCheckpointLineage(
            manifest: manifest, targetSpec: targetSpec,
            parentDirectory: parent,
            candidateDirectory: candidate,
            requirePreparedCandidatePath: true)
        let candidateFingerprint = validated.fingerprint
        let candidateMetadata = validated.metadata

        let suitePlans = resolvedSuites(
            manifest: manifest, targetSpec: targetSpec)
        let primarySuite = suitePlans[0]
        let plan = primarySuite.legacyPlan
        guard evaluationReportPaths.count == plan.evaluationSeeds.count else {
            throw invalid("publication requires exactly four declared reports")
        }
        var reportsBySeed = [UInt64: (PPOEvaluationMetrics, Data)]()
        for path in evaluationReportPaths {
            let data = try requiredData(URL(fileURLWithPath: path))
            let report = try JSONDecoder().decode(
                PPOEvaluationMetrics.self, from: data)
            guard reportsBySeed[report.evaluationSeed] == nil else {
                throw invalid("evaluation report seeds must be distinct")
            }
            try validate(
                report: report, targetSpec: targetSpec,
                criteria: liveCriteria(primarySuite.evaluationCriteria),
                expectedCheckpointDirectory:
                    manifest.candidateCheckpointDirectory,
                candidateFingerprint: candidateFingerprint,
                candidateMetadata: candidateMetadata,
                plan: plan,
                evaluationTaskConfiguration:
                    primarySuite.evaluationTaskConfiguration)
            reportsBySeed[report.evaluationSeed] = (report, data)
        }
        guard Set(reportsBySeed.keys) == Set(plan.evaluationSeeds) else {
            throw invalid("evaluation reports do not match the predeclared seeds")
        }
        let orderedReports = plan.evaluationSeeds.sorted().map {
            reportsBySeed[$0]!.0
        }

        let aggregateData = try requiredData(URL(fileURLWithPath: aggregatePath))
        let aggregate = try JSONDecoder().decode(
            PPOCheckpointEvaluationAggregate.self, from: aggregateData)
        try validateAggregate(
            aggregate, reports: orderedReports, targetSpec: targetSpec,
            candidateFingerprint: candidateFingerprint, plan: plan,
            evaluationTaskConfiguration:
                primarySuite.evaluationTaskConfiguration)

        let expectedAdditionalSuites = Array(suitePlans.dropFirst())
        guard additionalSuiteInputs.count == expectedAdditionalSuites.count,
              Set(additionalSuiteInputs.map(\.id)).count
                == additionalSuiteInputs.count else {
            throw invalid(
                "publication requires exactly the predeclared additional suites")
        }
        var additionalInputsByID = Dictionary(
            uniqueKeysWithValues: additionalSuiteInputs.map { ($0.id, $0) })
        var additionalQualifications = [
            String: (reports: [UInt64: (PPOEvaluationMetrics, Data)],
                     aggregate: PPOCheckpointEvaluationAggregate,
                     aggregateData: Data)
        ]()
        for suite in expectedAdditionalSuites {
            guard let input = additionalInputsByID.removeValue(forKey: suite.id)
            else {
                throw invalid("missing qualification suite \(suite.id)")
            }
            let suitePlan = suite.legacyPlan
            guard input.evaluationReportPaths.count
                    == suitePlan.evaluationSeeds.count else {
                throw invalid(
                    "suite \(suite.id) requires exactly four declared reports")
            }
            var suiteReports = [UInt64: (PPOEvaluationMetrics, Data)]()
            for path in input.evaluationReportPaths {
                let data = try requiredData(URL(fileURLWithPath: path))
                let report = try JSONDecoder().decode(
                    PPOEvaluationMetrics.self, from: data)
                guard suiteReports[report.evaluationSeed] == nil else {
                    throw invalid(
                        "suite \(suite.id) evaluation seeds must be distinct")
                }
                try validate(
                    report: report, targetSpec: targetSpec,
                    criteria: liveCriteria(suite.evaluationCriteria),
                    expectedCheckpointDirectory:
                        manifest.candidateCheckpointDirectory,
                    candidateFingerprint: candidateFingerprint,
                    candidateMetadata: candidateMetadata,
                    plan: suitePlan,
                    evaluationTaskConfiguration:
                        suite.evaluationTaskConfiguration)
                suiteReports[report.evaluationSeed] = (report, data)
            }
            guard Set(suiteReports.keys) == Set(suitePlan.evaluationSeeds) else {
                throw invalid(
                    "suite \(suite.id) reports do not match predeclared seeds")
            }
            let ordered = suitePlan.evaluationSeeds.sorted().map {
                suiteReports[$0]!.0
            }
            let aggregateData = try requiredData(
                URL(fileURLWithPath: input.aggregatePath))
            let suiteAggregate = try JSONDecoder().decode(
                PPOCheckpointEvaluationAggregate.self, from: aggregateData)
            try validateAggregate(
                suiteAggregate, reports: ordered, targetSpec: targetSpec,
                candidateFingerprint: candidateFingerprint, plan: suitePlan,
                evaluationTaskConfiguration:
                    suite.evaluationTaskConfiguration)
            additionalQualifications[suite.id] = (
                suiteReports, suiteAggregate, aggregateData)
        }
        guard additionalInputsByID.isEmpty else {
            throw invalid("publication includes an undeclared qualification suite")
        }
        var aggregatesBySuite = [primarySuite.id: aggregate]
        for suite in expectedAdditionalSuites {
            aggregatesBySuite[suite.id] = additionalQualifications[suite.id]!
                .aggregate
        }
        try validateComparisons(
            manifest.qualificationMatrix?.comparisons ?? [],
            aggregatesBySuite: aggregatesBySuite)

        let outputParent = output.deletingLastPathComponent()
        try manager.createDirectory(
            at: outputParent, withIntermediateDirectories: true)
        let staging = outputParent.appendingPathComponent(
            ".\(output.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true)
        var published = false
        defer {
            if !published { try? manager.removeItem(at: staging) }
        }
        _ = try VectorPolicyDeploymentBundle.export(
            checkpointDirectory: candidate.path,
            outputDirectory: staging.path)
        let qualificationDirectory = staging.appendingPathComponent(
            "qualification", isDirectory: true)
        try manager.createDirectory(
            at: qualificationDirectory, withIntermediateDirectories: false)

        let primaryDirectory: URL
        let primaryRelativeDirectory: String
        if manifest.schemaVersion == 2 {
            primaryDirectory = qualificationDirectory.appendingPathComponent(
                primarySuite.id, isDirectory: true)
            try manager.createDirectory(
                at: primaryDirectory, withIntermediateDirectories: false)
            primaryRelativeDirectory = "qualification/\(primarySuite.id)"
        } else {
            primaryDirectory = qualificationDirectory
            primaryRelativeDirectory = "qualification"
        }

        var reportEvidence = [VectorPolicyRequalificationReportEvidence]()
        for seed in plan.evaluationSeeds.sorted() {
            let data = reportsBySeed[seed]!.1
            let name = "eval-seed-\(seed).json"
            try data.write(
                to: primaryDirectory.appendingPathComponent(name),
                options: .atomic)
            reportEvidence.append(.init(
                evaluationSeed: seed,
                file: "\(primaryRelativeDirectory)/\(name)",
                sha256: sha256(data)))
        }
        let aggregateName = "aggregate.json"
        try aggregateData.write(
            to: primaryDirectory.appendingPathComponent(aggregateName),
            options: .atomic)
        var additionalSuiteEvidence = [
            VectorPolicyRequalificationSuiteEvidence
        ]()
        for suite in expectedAdditionalSuites {
            let validated = additionalQualifications[suite.id]!
            let suiteDirectory = qualificationDirectory.appendingPathComponent(
                suite.id, isDirectory: true)
            try manager.createDirectory(
                at: suiteDirectory, withIntermediateDirectories: false)
            var evidence = [VectorPolicyRequalificationReportEvidence]()
            for seed in suite.evaluationSeeds.sorted() {
                let data = validated.reports[seed]!.1
                let name = "eval-seed-\(seed).json"
                try data.write(
                    to: suiteDirectory.appendingPathComponent(name),
                    options: .atomic)
                evidence.append(.init(
                    evaluationSeed: seed,
                    file: "qualification/\(suite.id)/\(name)",
                    sha256: sha256(data)))
            }
            try validated.aggregateData.write(
                to: suiteDirectory.appendingPathComponent(aggregateName),
                options: .atomic)
            additionalSuiteEvidence.append(.init(
                id: suite.id, reports: evidence,
                aggregate: .init(
                    file: "qualification/\(suite.id)/\(aggregateName)",
                    sha256: sha256(validated.aggregateData))))
        }
        manifest.qualification = VectorPolicyRequalificationEvidence(
            reports: reportEvidence,
            aggregate: .init(
                file: "\(primaryRelativeDirectory)/\(aggregateName)",
                sha256: sha256(aggregateData)),
            additionalSuites: additionalSuiteEvidence.isEmpty
                ? nil : additionalSuiteEvidence)
        try encoded(manifest).write(
            to: staging.appendingPathComponent(manifestFileName),
            options: .atomic)
        guard try VectorPPOTrainer.checkpointFingerprint(directory: staging.path)
                == candidateFingerprint else {
            throw invalid("publication changed checkpoint identity")
        }
        _ = try verify(
            targetSpec: targetSpec, evaluationCriteria: evaluationCriteria,
            bundleDirectory: staging.path,
            parentCheckpointDirectory: parentCheckpointDirectory,
            suiteContracts: suiteContracts)
        try manager.moveItem(at: staging, to: output)
        published = true
        return manifest
    }

    /// Verify a sealed bundle's zero-update lineage and every tracked evidence
    /// byte. The ordinary checkpoint fingerprint intentionally excludes
    /// reports, so accepted requalification bundles must run this verifier in
    /// addition to generic policy inference verification.
    @discardableResult
    public static func verify(
        targetSpec: RLTaskSpec,
        evaluationCriteria: RLEvaluationCriteria,
        bundleDirectory: String,
        parentCheckpointDirectory: String,
        suiteContracts: [VectorPolicyRequalificationSuiteContract] = []
    ) throws -> VectorPolicyRequalificationManifest {
        let bundle = URL(fileURLWithPath: bundleDirectory, isDirectory: true)
        let parent = URL(
            fileURLWithPath: parentCheckpointDirectory, isDirectory: true)
        let manifestData = try requiredData(
            bundle.appendingPathComponent(manifestFileName))
        let manifest = try JSONDecoder().decode(
            VectorPolicyRequalificationManifest.self, from: manifestData)
        guard [1, 2].contains(manifest.schemaVersion),
              let evidence = manifest.qualification else {
            throw invalid("requalification bundle is not sealed")
        }
        try validate(plan: manifest.qualificationPlan)
        try validate(criteria: manifest.evaluationCriteria)
        try validate(
            matrix: manifest.qualificationMatrix,
            primaryPlan: manifest.qualificationPlan,
            primaryCriteria: manifest.evaluationCriteria,
            targetSpec: targetSpec, suiteContracts: suiteContracts)
        guard manifest.schemaVersion
                == (manifest.qualificationMatrix == nil ? 1 : 2) else {
            throw invalid("requalification schema does not match its matrix")
        }
        try validateManifestContract(manifest, targetSpec: targetSpec)
        guard manifest.evaluationCriteria
                == VectorPolicyRequalificationCriteria(evaluationCriteria) else {
            throw invalid(
                "sealed evaluation criteria do not match the current task")
        }

        let validated = try validateCheckpointLineage(
            manifest: manifest, targetSpec: targetSpec,
            parentDirectory: parent, candidateDirectory: bundle,
            requirePreparedCandidatePath: false)
        let deployment = try JSONDecoder().decode(
            VectorPolicyDeploymentManifest.self,
            from: requiredData(bundle.appendingPathComponent(
                VectorPolicyDeploymentBundle.manifestFileName)))
        guard let architectureVersion = validated.metadata.architectureVersion else {
            throw invalid("requalified metadata has no architecture version")
        }
        let expectedNormalizationClip: Float? = validated.metadata.ppo
            .normalizeObservations ? 10 : nil
        guard deployment.schemaVersion == 1,
              deployment.task == targetSpec.id,
              deployment.taskRevision == targetSpec.revision,
              deployment.checkpointFingerprint == validated.fingerprint,
              deployment.policySHA256 == manifest.candidatePolicySHA256,
              deployment.policyFile == "policy.safetensors",
              deployment.metadataFile == "metadata.json",
              deployment.trainingStateFile == "training-state.json",
              deployment.architectureVersion == architectureVersion,
              deployment.observationDimension
                == validated.metadata.observationDimension,
              deployment.actionDimension == validated.metadata.actionDimension,
              deployment.simulationStepSeconds
                == validated.metadata.simulationStep,
              deployment.controlDecimation
                == validated.metadata.controlDecimation,
              deployment.controlFrequencyHz
                == 1 / (validated.metadata.simulationStep
                    * Float(validated.metadata.controlDecimation)),
              deployment.normalizesObservations
                == validated.metadata.ppo.normalizeObservations,
              deployment.observationNormalizationClip
                == expectedNormalizationClip,
              deployment.actionDistribution
                == validated.metadata.ppo.resolvedActionDistribution.rawValue,
              deployment.taskConfiguration == targetSpec.configurationValues,
              deployment.trainingUpdates
                == validated.trainingState.completedUpdates,
              deployment.trainingEnvironmentSteps
                == validated.trainingState.environmentSteps else {
            throw invalid("deployment manifest does not match requalification")
        }

        let suitePlans = resolvedSuites(
            manifest: manifest, targetSpec: targetSpec)
        let primarySuite = suitePlans[0]
        let plan = primarySuite.legacyPlan
        guard evidence.reports.count == plan.evaluationSeeds.count else {
            throw invalid("sealed report evidence count is invalid")
        }
        guard evidence.reports.map(\.evaluationSeed)
                == plan.evaluationSeeds else {
            throw invalid("sealed primary report order is not canonical")
        }
        var reportsBySeed = [UInt64: PPOEvaluationMetrics]()
        for item in evidence.reports {
            guard reportsBySeed[item.evaluationSeed] == nil else {
                throw invalid("sealed report evidence has duplicate seeds")
            }
            if manifest.schemaVersion == 2 {
                let expectedPath =
                    "qualification/\(primarySuite.id)/eval-seed-"
                    + "\(item.evaluationSeed).json"
                guard item.file == expectedPath else {
                    throw invalid(
                        "sealed primary report path is not canonical")
                }
            }
            let data = try verifiedEvidenceData(
                root: bundle, relativePath: item.file,
                expectedSHA256: item.sha256)
            let report = try JSONDecoder().decode(
                PPOEvaluationMetrics.self, from: data)
            guard report.evaluationSeed == item.evaluationSeed else {
                throw invalid("sealed report seed does not match its manifest")
            }
            try validate(
                report: report, targetSpec: targetSpec,
                criteria: liveCriteria(primarySuite.evaluationCriteria),
                expectedCheckpointDirectory:
                    manifest.candidateCheckpointDirectory,
                candidateFingerprint: validated.fingerprint,
                candidateMetadata: validated.metadata,
                plan: plan,
                evaluationTaskConfiguration:
                    primarySuite.evaluationTaskConfiguration)
            reportsBySeed[item.evaluationSeed] = report
        }
        guard Set(reportsBySeed.keys) == Set(plan.evaluationSeeds) else {
            throw invalid("sealed reports do not match the predeclared seeds")
        }
        let orderedReports = plan.evaluationSeeds.sorted().map {
            reportsBySeed[$0]!
        }
        if manifest.schemaVersion == 2 {
            guard evidence.aggregate.file
                    == "qualification/\(primarySuite.id)/aggregate.json" else {
                throw invalid("sealed primary aggregate path is not canonical")
            }
        }
        let aggregateData = try verifiedEvidenceData(
            root: bundle, relativePath: evidence.aggregate.file,
            expectedSHA256: evidence.aggregate.sha256)
        let aggregate = try JSONDecoder().decode(
            PPOCheckpointEvaluationAggregate.self, from: aggregateData)
        try validateAggregate(
            aggregate, reports: orderedReports, targetSpec: targetSpec,
            candidateFingerprint: validated.fingerprint, plan: plan,
            evaluationTaskConfiguration:
                primarySuite.evaluationTaskConfiguration)

        let expectedAdditionalSuites = Array(suitePlans.dropFirst())
        if manifest.schemaVersion == 1,
           evidence.additionalSuites != nil {
            throw invalid("schema-v1 evidence must not contain suite extensions")
        }
        let additionalEvidence = evidence.additionalSuites ?? []
        guard additionalEvidence.count == expectedAdditionalSuites.count,
              Set(additionalEvidence.map(\.id)).count
                == additionalEvidence.count,
              additionalEvidence.map(\.id)
                == expectedAdditionalSuites.map(\.id) else {
            throw invalid(
                "sealed evidence does not match the predeclared suite matrix")
        }
        var additionalEvidenceByID = Dictionary(
            uniqueKeysWithValues: additionalEvidence.map { ($0.id, $0) })
        var aggregatesBySuite = [primarySuite.id: aggregate]
        for suite in expectedAdditionalSuites {
            guard let suiteEvidence = additionalEvidenceByID.removeValue(
                forKey: suite.id) else {
                throw invalid("sealed evidence is missing suite \(suite.id)")
            }
            let suitePlan = suite.legacyPlan
            guard suiteEvidence.reports.count
                    == suitePlan.evaluationSeeds.count else {
                throw invalid("sealed suite \(suite.id) report count is invalid")
            }
            guard suiteEvidence.reports.map(\.evaluationSeed)
                    == suitePlan.evaluationSeeds else {
                throw invalid(
                    "sealed suite \(suite.id) report order is not canonical")
            }
            var suiteReportsBySeed = [UInt64: PPOEvaluationMetrics]()
            for item in suiteEvidence.reports {
                guard suiteReportsBySeed[item.evaluationSeed] == nil else {
                    throw invalid(
                        "sealed suite \(suite.id) has duplicate report seeds")
                }
                let expectedPath = "qualification/\(suite.id)/eval-seed-"
                    + "\(item.evaluationSeed).json"
                guard item.file == expectedPath else {
                    throw invalid(
                        "sealed suite \(suite.id) report path is not canonical")
                }
                let data = try verifiedEvidenceData(
                    root: bundle, relativePath: item.file,
                    expectedSHA256: item.sha256)
                let report = try JSONDecoder().decode(
                    PPOEvaluationMetrics.self, from: data)
                guard report.evaluationSeed == item.evaluationSeed else {
                    throw invalid(
                        "sealed suite \(suite.id) report seed is inconsistent")
                }
                try validate(
                    report: report, targetSpec: targetSpec,
                    criteria: liveCriteria(suite.evaluationCriteria),
                    expectedCheckpointDirectory:
                        manifest.candidateCheckpointDirectory,
                    candidateFingerprint: validated.fingerprint,
                    candidateMetadata: validated.metadata,
                    plan: suitePlan,
                    evaluationTaskConfiguration:
                        suite.evaluationTaskConfiguration)
                suiteReportsBySeed[item.evaluationSeed] = report
            }
            guard Set(suiteReportsBySeed.keys)
                    == Set(suitePlan.evaluationSeeds) else {
                throw invalid(
                    "sealed suite \(suite.id) reports do not match its seeds")
            }
            let expectedAggregatePath =
                "qualification/\(suite.id)/aggregate.json"
            guard suiteEvidence.aggregate.file == expectedAggregatePath else {
                throw invalid(
                    "sealed suite \(suite.id) aggregate path is not canonical")
            }
            let aggregateData = try verifiedEvidenceData(
                root: bundle,
                relativePath: suiteEvidence.aggregate.file,
                expectedSHA256: suiteEvidence.aggregate.sha256)
            let suiteAggregate = try JSONDecoder().decode(
                PPOCheckpointEvaluationAggregate.self, from: aggregateData)
            let ordered = suitePlan.evaluationSeeds.sorted().map {
                suiteReportsBySeed[$0]!
            }
            try validateAggregate(
                suiteAggregate, reports: ordered, targetSpec: targetSpec,
                candidateFingerprint: validated.fingerprint, plan: suitePlan,
                evaluationTaskConfiguration:
                    suite.evaluationTaskConfiguration)
            aggregatesBySuite[suite.id] = suiteAggregate
        }
        guard additionalEvidenceByID.isEmpty else {
            throw invalid("sealed evidence includes an undeclared suite")
        }
        try validateComparisons(
            manifest.qualificationMatrix?.comparisons ?? [],
            aggregatesBySuite: aggregatesBySuite)
        if manifest.schemaVersion == 2 {
            try validateSchema2Inventory(
                bundle: bundle, suitePlans: suitePlans)
        }
        return manifest
    }

    private struct ValidatedCheckpointLineage {
        var metadata: VectorPolicyMetadata
        var trainingState: VectorPPOTrainingState
        var fingerprint: String
    }

    /// Re-derive every permitted target-side byte from the parent. Manifest
    /// hashes alone are insufficient because a tamperer could update both a
    /// file and its claimed digest.
    private static func validateCheckpointLineage(
        manifest: VectorPolicyRequalificationManifest,
        targetSpec: RLTaskSpec,
        parentDirectory: URL,
        candidateDirectory: URL,
        requirePreparedCandidatePath: Bool
    ) throws -> ValidatedCheckpointLineage {
        guard standardizedPath(parentDirectory.path)
                == standardizedPath(manifest.parentCheckpointDirectory),
              !requirePreparedCandidatePath
                || standardizedPath(candidateDirectory.path)
                    == standardizedPath(
                        manifest.candidateCheckpointDirectory) else {
            throw invalid("checkpoint lineage paths do not match the manifest")
        }

        let parentFingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: parentDirectory.path)
        let candidateFingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: candidateDirectory.path)
        guard parentFingerprint == manifest.parentCheckpointFingerprint,
              candidateFingerprint == manifest.candidateCheckpointFingerprint,
              parentFingerprint != candidateFingerprint else {
            throw invalid("parent or candidate checkpoint identity is invalid")
        }

        let parentPolicyData = try requiredData(
            parentDirectory.appendingPathComponent("policy.safetensors"))
        let parentMetadataData = try requiredData(
            parentDirectory.appendingPathComponent("metadata.json"))
        let parentStateData = try requiredData(
            parentDirectory.appendingPathComponent("training-state.json"))
        let candidatePolicyData = try requiredData(
            candidateDirectory.appendingPathComponent("policy.safetensors"))
        let candidateMetadataData = try requiredData(
            candidateDirectory.appendingPathComponent("metadata.json"))
        let candidateStateData = try requiredData(
            candidateDirectory.appendingPathComponent("training-state.json"))
        guard parentPolicyData == candidatePolicyData,
              manifest.parentPolicySHA256
                == manifest.candidatePolicySHA256,
              sha256(parentPolicyData) == manifest.parentPolicySHA256,
              sha256(candidatePolicyData) == manifest.candidatePolicySHA256,
              sha256(parentMetadataData) == manifest.parentMetadataSHA256,
              sha256(candidateMetadataData) == manifest.candidateMetadataSHA256,
              sha256(parentStateData) == manifest.parentTrainingStateSHA256,
              sha256(candidateStateData)
                == manifest.candidateTrainingStateSHA256 else {
            throw invalid("checkpoint lineage files do not match the manifest")
        }

        let decoder = JSONDecoder()
        let parentMetadata = try decoder.decode(
            VectorPolicyMetadata.self, from: parentMetadataData)
        let parentState = try decoder.decode(
            VectorPPOTrainingState.self, from: parentStateData)
        let candidateMetadata = try decoder.decode(
            VectorPolicyMetadata.self, from: candidateMetadataData)
        let candidateState = try decoder.decode(
            VectorPPOTrainingState.self, from: candidateStateData)
        let expectedMetadata = expectedCandidateMetadata(
            parent: parentMetadata, targetSpec: targetSpec,
            inferenceBatchSize: manifest.inferenceBatchSize,
            parentCheckpointDirectory: manifest.parentCheckpointDirectory)
        let expectedState = zeroUpdateTrainingState()
        let parentMismatches = parentMetadata.compatibilityMismatches(
            with: targetSpec)
        guard parentMismatches.count == 1,
              parentMismatches[0].hasPrefix("revision "),
              try canonicalJSON(candidateMetadata)
                == canonicalJSON(expectedMetadata),
              try canonicalJSON(candidateState) == canonicalJSON(expectedState),
              candidateMetadata.compatibilityMismatches(with: targetSpec).isEmpty,
              manifest.sourceTaskRevision == parentMetadata.taskRevision,
              manifest.targetTaskRevision == candidateMetadata.taskRevision,
              manifest.parentTrainingUpdates == parentState.completedUpdates,
              manifest.parentTrainingEnvironmentSteps
                == parentState.environmentSteps,
              manifest.targetTrainingUpdates == candidateState.completedUpdates,
              manifest.targetTrainingEnvironmentSteps
                == candidateState.environmentSteps else {
            throw invalid("candidate is not the exact permitted zero-update transform")
        }
        return .init(
            metadata: candidateMetadata, trainingState: candidateState,
            fingerprint: candidateFingerprint)
    }

    private static func validateAggregate(
        _ aggregate: PPOCheckpointEvaluationAggregate,
        reports: [PPOEvaluationMetrics],
        targetSpec: RLTaskSpec,
        candidateFingerprint: String,
        plan: VectorPolicyRequalificationPlan,
        evaluationTaskConfiguration: [String: Float]
    ) throws {
        try aggregate.validateStructure()
        let recomputed = try PPOCheckpointEvaluationAggregate.make(reports)
        let transferred = evaluationTaskConfiguration
            != targetSpec.configurationValues
        guard try canonicalJSON(aggregate) == canonicalJSON(recomputed),
              aggregate.scope == "single_checkpoint_across_evaluation_seeds",
              aggregate.task == targetSpec.id,
              aggregate.taskRevision == targetSpec.revision,
              aggregate.checkpointFingerprint == candidateFingerprint,
              aggregate.evaluationTaskConfiguration
                == evaluationTaskConfiguration,
              aggregate.taskConfigurationTransferred == transferred,
              aggregate.evaluationSeeds == plan.evaluationSeeds.sorted(),
              aggregate.evaluationEnvironments == plan.evaluationEnvironments,
              aggregate.runs == 4,
              aggregate.requiredRuns == 4,
              aggregate.requiredEpisodesPerRun == plan.episodesPerReport,
              aggregate.totalEpisodes == 4 * plan.episodesPerReport,
              aggregate.acceptedRuns == 4,
              aggregate.allRunsPassed,
              aggregate.hasRequiredRunCount,
              aggregate.allRunsHaveRequiredEpisodes,
              aggregate.provenanceComplete,
              aggregate.robustAcrossEvaluationSeeds else {
            throw invalid(
                "aggregate is not the exact robust result of the declared reports")
        }
    }

    private static func validate(
        report: PPOEvaluationMetrics,
        targetSpec: RLTaskSpec,
        criteria: RLEvaluationCriteria,
        expectedCheckpointDirectory: String,
        candidateFingerprint: String,
        candidateMetadata: VectorPolicyMetadata,
        plan: VectorPolicyRequalificationPlan,
        evaluationTaskConfiguration: [String: Float]
    ) throws {
        try report.validateStructure()
        let failures = criteria.failures(
            successRate: report.successRate,
            meanEpisodeLength: report.meanEpisodeLength,
            maxEpisodeSteps: targetSpec.maxEpisodeSteps,
            taskMetrics: report.taskMetrics)
        let derivedSuccessRate = Float(report.successes)
            / Float(max(report.episodes, 1))
        let transferred = evaluationTaskConfiguration
            != targetSpec.configurationValues
        guard (report.provenanceVersion ?? 0) >= 3,
              report.task == targetSpec.id,
              report.taskRevision == targetSpec.revision,
              report.checkpointTaskConfiguration
                == targetSpec.configurationValues,
              report.evaluationTaskConfiguration
                == evaluationTaskConfiguration,
              report.taskConfigurationTransferred == transferred,
              standardizedPath(report.checkpointDirectory)
                == standardizedPath(expectedCheckpointDirectory),
              report.checkpointFingerprint == candidateFingerprint,
              report.initializationCheckpoint
                == candidateMetadata.ppo.initializationCheckpoint,
              report.trainingSeed == candidateMetadata.ppo.seed,
              report.trainingUpdates == 0,
              report.trainingEnvironmentSteps == 0,
              report.evaluationEnvironments == plan.evaluationEnvironments,
              report.episodes == plan.episodesPerReport,
              report.successes >= 0,
              report.successes <= report.episodes,
              report.successRate.isFinite,
              abs(report.successRate - derivedSuccessRate) <= 1e-6,
              report.meanReturn.isFinite,
              report.meanEpisodeLength.isFinite,
              report.meanEpisodeLength >= 0,
              report.meanEpisodeLength <= Float(targetSpec.maxEpisodeSteps),
              report.taskMetrics.values.allSatisfy(\.isFinite),
              report.acceptance?.passed == true,
              report.acceptance?.failures.isEmpty == true,
              failures.isEmpty else {
            throw invalid(
                "evaluation seed \(report.evaluationSeed) does not satisfy the "
                    + "declared target contract and task-owned acceptance gate")
        }
    }

    private static func validate(
        plan: VectorPolicyRequalificationPlan
    ) throws {
        guard plan.evaluationSeeds.count == 4,
              Set(plan.evaluationSeeds).count == 4,
              plan.evaluationSeeds == plan.evaluationSeeds.sorted(),
              plan.evaluationEnvironments > 0,
              plan.episodesPerReport == 512 else {
            throw invalid(
                "requalification requires four distinct seeds, a positive "
                    + "replica count, and exactly 512 episodes per report")
        }
    }

    private static func validate(
        matrix: VectorPolicyRequalificationMatrix?,
        primaryPlan: VectorPolicyRequalificationPlan,
        primaryCriteria: VectorPolicyRequalificationCriteria,
        targetSpec: RLTaskSpec,
        suiteContracts: [VectorPolicyRequalificationSuiteContract]
    ) throws {
        guard let matrix else {
            guard suiteContracts.isEmpty else {
                throw invalid(
                    "schema-v1 requalification must not supply suite contracts")
            }
            return
        }
        guard matrix.suites.count >= 2,
              !matrix.comparisons.isEmpty else {
            throw invalid(
                "multi-suite requalification requires at least two suites "
                    + "and one cross-suite comparison")
        }
        let identifiers = matrix.suites.map(\.id)
        guard Set(identifiers).count == identifiers.count,
              suiteContracts.count == matrix.suites.count,
              Set(suiteContracts.map(\.id)).count == suiteContracts.count,
              Set(suiteContracts.map(\.id)) == Set(identifiers) else {
            throw invalid("qualification suite identifiers must be distinct")
        }
        let contractsByID = Dictionary(
            uniqueKeysWithValues: suiteContracts.map { ($0.id, $0) })
        var allSeeds = Set<UInt64>()
        for (index, suite) in matrix.suites.enumerated() {
            try validate(plan: suite.legacyPlan)
            try validate(criteria: suite.evaluationCriteria)
            guard let contract = contractsByID[suite.id] else {
                throw invalid("qualification suite contract is missing")
            }
            let spec = contract.taskSpec
            guard spec.id == targetSpec.id,
                  spec.revision == targetSpec.revision,
                  spec.observation == targetSpec.observation,
                  spec.privilegedObservation
                    == targetSpec.privilegedObservation,
                  spec.action == targetSpec.action,
                  spec.maxEpisodeSteps == targetSpec.maxEpisodeSteps,
                  spec.simulationStep == targetSpec.simulationStep,
                  spec.controlDecimation == targetSpec.controlDecimation,
                  spec.numEnvironments == suite.evaluationEnvironments,
                  spec.autoReset == false,
                  spec.configurationValues
                    == suite.evaluationTaskConfiguration,
                  VectorPolicyRequalificationCriteria(
                    contract.evaluationCriteria)
                    == suite.evaluationCriteria else {
                throw invalid(
                    "qualification suite \(suite.id) does not match its "
                        + "live task contract")
            }
            guard validSuiteIdentifier(suite.id),
                  suite.evaluationTaskConfiguration.values.allSatisfy(
                    \.isFinite),
                  suite.changedConfigurationFields
                    == suite.changedConfigurationFields.sorted(),
                  Set(suite.changedConfigurationFields).count
                    == suite.changedConfigurationFields.count else {
                throw invalid("qualification suite \(suite.id) is invalid")
            }
            let changedFields = Set(
                targetSpec.configurationValues.keys.filter {
                    targetSpec.configurationValues[$0]
                        != suite.evaluationTaskConfiguration[$0]
                } + suite.evaluationTaskConfiguration.keys.filter {
                    targetSpec.configurationValues[$0]
                        != suite.evaluationTaskConfiguration[$0]
                })
            guard suite.changedConfigurationFields
                    == changedFields.sorted() else {
                throw invalid(
                    "qualification suite \(suite.id) does not declare its "
                        + "exact task-configuration delta")
            }
            if index == 0 {
                guard suite.legacyPlan == primaryPlan,
                      suite.evaluationTaskConfiguration
                        == targetSpec.configurationValues,
                      suite.changedConfigurationFields.isEmpty,
                      suite.evaluationCriteria == primaryCriteria else {
                    throw invalid(
                        "the first qualification suite must be the nominal "
                            + "checkpoint contract and mirror qualificationPlan")
                }
            } else {
                guard suite.evaluationEnvironments
                        == primaryPlan.evaluationEnvironments,
                      !suite.changedConfigurationFields.isEmpty else {
                    throw invalid(
                        "each validation suite must change at least one "
                            + "declared task option and use the primary batch size")
                }
            }
            for seed in suite.evaluationSeeds {
                guard allSeeds.insert(seed).inserted else {
                    throw invalid(
                        "qualification seeds must not overlap across suites")
                }
            }
        }
        let identifiersSet = Set(identifiers)
        guard matrix.comparisons.count == matrix.suites.count - 1 else {
            throw invalid(
                "each validation suite requires exactly one comparison")
        }
        var comparisonPairs = Set<String>()
        for comparison in matrix.comparisons {
            let key = comparison.baselineSuite + "\u{0}"
                + comparison.evaluatedSuite
            guard identifiersSet.contains(comparison.baselineSuite),
                  identifiersSet.contains(comparison.evaluatedSuite),
                  comparison.baselineSuite == identifiers[0],
                  comparison.evaluatedSuite != identifiers[0],
                  comparison.maximumPooledSuccessRateDrop.isFinite,
                  (0...1).contains(
                    comparison.maximumPooledSuccessRateDrop),
                  comparisonPairs.insert(key).inserted else {
                throw invalid("qualification suite comparison is invalid")
            }
        }
        let evaluatedSuites = Set(matrix.comparisons.map(\.evaluatedSuite))
        guard Set(identifiers.dropFirst()) == evaluatedSuites else {
            throw invalid(
                "every validation suite requires a cross-suite comparison")
        }
    }

    private static func resolvedSuites(
        manifest: VectorPolicyRequalificationManifest,
        targetSpec: RLTaskSpec
    ) -> [VectorPolicyRequalificationSuitePlan] {
        if let matrix = manifest.qualificationMatrix {
            return matrix.suites
        }
        return [.init(
            id: "nominal",
            evaluationSeeds: manifest.qualificationPlan.evaluationSeeds,
            evaluationEnvironments:
                manifest.qualificationPlan.evaluationEnvironments,
            episodesPerReport:
                manifest.qualificationPlan.episodesPerReport,
            evaluationTaskConfiguration: targetSpec.configurationValues,
            changedConfigurationFields: [],
            evaluationCriteria: liveCriteria(manifest.evaluationCriteria))]
    }

    private static func liveCriteria(
        _ criteria: VectorPolicyRequalificationCriteria
    ) -> RLEvaluationCriteria {
        .init(
            minimumSuccessRate: criteria.minimumSuccessRate,
            minimumMeanEpisodeLengthFraction:
                criteria.minimumMeanEpisodeLengthFraction,
            minimumTaskMetrics: criteria.minimumTaskMetrics,
            maximumTaskMetrics: criteria.maximumTaskMetrics)
    }

    private static func validateComparisons(
        _ comparisons: [VectorPolicyRequalificationComparison],
        aggregatesBySuite: [String: PPOCheckpointEvaluationAggregate]
    ) throws {
        for comparison in comparisons {
            guard let baseline = aggregatesBySuite[comparison.baselineSuite],
                  let evaluated = aggregatesBySuite[comparison.evaluatedSuite]
            else {
                throw invalid("qualification comparison suite is missing")
            }
            let drop = baseline.pooledSuccessRate
                - evaluated.pooledSuccessRate
            guard drop <= comparison.maximumPooledSuccessRateDrop else {
                throw invalid(
                    "suite \(comparison.evaluatedSuite) pooled success rate "
                        + "degraded beyond its predeclared bound")
            }
        }
    }

    private static func validSuiteIdentifier(_ value: String) -> Bool {
        guard let first = value.utf8.first,
              (97...122).contains(first) || (48...57).contains(first) else {
            return false
        }
        return value.utf8.allSatisfy {
            (97...122).contains($0) || (48...57).contains($0) || $0 == 45
        }
    }

    private static func validateSchema2Inventory(
        bundle: URL, suitePlans: [VectorPolicyRequalificationSuitePlan]
    ) throws {
        let manager = FileManager.default
        let expectedTopLevel = Set([
            "metadata.json", "policy.safetensors", "training-state.json",
            VectorPolicyDeploymentBundle.manifestFileName, manifestFileName,
            "qualification",
        ])
        let topLevel = try manager.contentsOfDirectory(
            atPath: bundle.path)
        let actualTopLevel = Set(topLevel)
        guard actualTopLevel == expectedTopLevel
                || actualTopLevel
                    == expectedTopLevel.union([PolicyBundleManifest.fileName]) else {
            throw invalid("schema-v2 bundle has missing or extra top-level entries")
        }
        for name in actualTopLevel {
            let url = bundle.appendingPathComponent(name)
            let values = try url.resourceValues(
                forKeys: [
                    .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
                ])
            let expectedDirectory = name == "qualification"
            guard values.isSymbolicLink != true,
                  expectedDirectory
                    ? values.isDirectory == true
                    : values.isRegularFile == true else {
                throw invalid("schema-v2 top-level inventory is invalid")
            }
        }
        var expectedFiles = Set<String>()
        var expectedDirectories = Set<String>()
        for suite in suitePlans {
            let prefix = "qualification/\(suite.id)"
            expectedDirectories.insert(prefix)
            expectedFiles.insert("\(prefix)/aggregate.json")
            for seed in suite.evaluationSeeds {
                expectedFiles.insert("\(prefix)/eval-seed-\(seed).json")
            }
        }
        guard let enumerator = manager.enumerator(
            at: bundle.appendingPathComponent("qualification"),
            includingPropertiesForKeys: [
                .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
            ])
        else {
            throw invalid("schema-v2 qualification directory is missing")
        }
        var actualFiles = Set<String>()
        var actualDirectories = Set<String>()
        let resolvedBundlePath = bundle.resolvingSymlinksInPath()
            .standardizedFileURL.path
        for case let url as URL in enumerator {
            let values = try url.resourceValues(
                forKeys: [
                    .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
                ])
            guard values.isSymbolicLink != true else {
                throw invalid("schema-v2 evidence must not contain symlinks")
            }
            if values.isRegularFile == true {
                let resolvedPath = url.resolvingSymlinksInPath()
                    .standardizedFileURL.path
                let prefix = resolvedBundlePath + "/"
                guard resolvedPath.hasPrefix(prefix) else {
                    throw invalid("schema-v2 evidence escaped its bundle")
                }
                actualFiles.insert(
                    String(resolvedPath.dropFirst(prefix.count)))
            } else if values.isDirectory == true {
                let resolvedPath = url.resolvingSymlinksInPath()
                    .standardizedFileURL.path
                let prefix = resolvedBundlePath + "/"
                guard resolvedPath.hasPrefix(prefix) else {
                    throw invalid("schema-v2 evidence escaped its bundle")
                }
                actualDirectories.insert(
                    String(resolvedPath.dropFirst(prefix.count)))
            } else {
                throw invalid("schema-v2 evidence contains a special file")
            }
        }
        guard actualFiles == expectedFiles,
              actualDirectories == expectedDirectories else {
            throw invalid("schema-v2 qualification has missing or extra files")
        }
    }

    private static func validate(
        criteria: VectorPolicyRequalificationCriteria
    ) throws {
        let overlappingMetrics = Set(criteria.minimumTaskMetrics.keys)
            .intersection(criteria.maximumTaskMetrics.keys)
        guard criteria.minimumSuccessRate.isFinite,
              (0...1).contains(criteria.minimumSuccessRate),
              criteria.minimumMeanEpisodeLengthFraction.isFinite,
              (0...1).contains(
                criteria.minimumMeanEpisodeLengthFraction),
              criteria.minimumTaskMetrics.values.allSatisfy(\.isFinite),
              criteria.maximumTaskMetrics.values.allSatisfy(\.isFinite),
              overlappingMetrics.allSatisfy({
                criteria.minimumTaskMetrics[$0]!
                    <= criteria.maximumTaskMetrics[$0]!
              }) else {
            throw invalid("requalification evaluation criteria are invalid")
        }
    }

    private static func validateManifestContract(
        _ manifest: VectorPolicyRequalificationManifest,
        targetSpec: RLTaskSpec
    ) throws {
        guard manifest.task == targetSpec.id,
              !manifest.parentCheckpointDirectory.isEmpty,
              !manifest.candidateCheckpointDirectory.isEmpty,
              standardizedPath(manifest.parentCheckpointDirectory)
                != standardizedPath(manifest.candidateCheckpointDirectory),
              manifest.targetTaskRevision == targetSpec.revision,
              manifest.sourceTaskRevision != manifest.targetTaskRevision,
              manifest.taskConfiguration == targetSpec.configurationValues,
              manifest.observationDimension
                == targetSpec.observation.elementCount,
              manifest.actionDimension == targetSpec.action.elementCount,
              manifest.simulationStepSeconds == targetSpec.simulationStep,
              manifest.controlDecimation == targetSpec.controlDecimation,
              manifest.maxEpisodeSteps == targetSpec.maxEpisodeSteps,
              manifest.inferenceBatchSize
                == manifest.qualificationPlan.evaluationEnvironments,
              manifest.inferenceBatchSize > 0,
              manifest.targetTrainingUpdates == 0,
              manifest.targetTrainingEnvironmentSteps == 0,
              manifest.changedFields == expectedChangedFields else {
            throw invalid("requalification manifest does not match the target task")
        }
        try validateDeclaredSourceCommit(manifest.declaredSourceCommit)
    }

    private static func validateDeclaredSourceCommit(_ value: String) throws {
        guard [40, 64].contains(value.count),
              value.utf8.allSatisfy({ byte in
                  (48...57).contains(byte) || (97...102).contains(byte)
              }) else {
            throw invalid(
                "declared source commit must be a full lowercase hexadecimal object ID")
        }
    }

    private static func expectedCandidateMetadata(
        parent: VectorPolicyMetadata,
        targetSpec: RLTaskSpec,
        inferenceBatchSize: Int,
        parentCheckpointDirectory: String
    ) -> VectorPolicyMetadata {
        var metadata = parent
        metadata.taskRevision = targetSpec.revision
        metadata.taskConfiguration = targetSpec.configurationValues
        metadata.inferenceBatchSize = inferenceBatchSize
        metadata.ppo.initializationCheckpoint = parentCheckpointDirectory
        metadata.ppo.updates = 0
        return metadata
    }

    private static func zeroUpdateTrainingState() -> VectorPPOTrainingState {
        VectorPPOTrainingState(
            completedUpdates: 0, environmentSteps: 0, optimizerSteps: 0,
            adaptiveLearningRate: nil)
    }

    private static func requireAbsent(
        _ url: URL, manager: FileManager
    ) throws {
        guard !manager.fileExists(atPath: url.path) else {
            throw invalid("output already exists: \(url.path)")
        }
    }

    private static func requiredData(_ url: URL) throws -> Data {
        let data = try Data(contentsOf: url)
        guard !data.isEmpty else {
            throw invalid("required file is empty: \(url.path)")
        }
        return data
    }

    private static func verifiedEvidenceData(
        root: URL, relativePath: String, expectedSHA256: String
    ) throws -> Data {
        let components = relativePath.split(
            separator: "/", omittingEmptySubsequences: false)
        guard !relativePath.hasPrefix("/"), !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != ".." && $0 != "." }),
              expectedSHA256.count == 64,
              expectedSHA256.utf8.allSatisfy({ byte in
                  (48...57).contains(byte) || (97...102).contains(byte)
              }) else {
            throw invalid("qualification evidence path or digest is invalid")
        }
        let url = root.appendingPathComponent(relativePath)
        let rootPath = root.resolvingSymlinksInPath().standardizedFileURL.path
        let evidencePath = url.resolvingSymlinksInPath()
            .standardizedFileURL.path
        guard evidencePath.hasPrefix(rootPath + "/") else {
            throw invalid("qualification evidence escapes its bundle")
        }
        let data = try requiredData(url)
        guard sha256(data) == expectedSHA256 else {
            throw invalid("qualification evidence SHA-256 does not match")
        }
        return data
    }

    private static func encoded<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(value)
    }

    private static func canonicalJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func standardizedPath(_ path: String) -> String {
        URL(fileURLWithPath: path).resolvingSymlinksInPath()
            .standardizedFileURL.path
    }

    private static func isSameOrDescendant(_ url: URL, of root: URL) -> Bool {
        let resolvedParent = url.deletingLastPathComponent()
            .resolvingSymlinksInPath().standardizedFileURL
        let path = resolvedParent.appendingPathComponent(url.lastPathComponent)
            .standardizedFileURL.path
        let rootPath = root.resolvingSymlinksInPath()
            .standardizedFileURL.path
        return path == rootPath || path.hasPrefix(rootPath + "/")
    }

    private static func invalid(_ message: String) -> RLEnvironmentError {
        .invalidConfiguration(message)
    }
}
