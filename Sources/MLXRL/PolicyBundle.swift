import CryptoKit
import Foundation
import RL

/// Portable, repository-independent policy replay artifact.
///
/// A bundle is an ordinary directory whose `policy-bundle.json` names every
/// runtime file relative to that directory. The manifest owns simulation and
/// presentation configuration; the application never infers those values
/// from a repository path or a policy identifier.
public struct PolicyBundleManifest: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1
    public static let fileName = "policy-bundle.json"

    public var schemaVersion: Int
    public var identifier: String
    public var title: String
    public var summary: String
    public var runtime: Runtime
    public var simulation: Simulation
    public var presentation: Presentation

    public init(
        schemaVersion: Int = currentSchemaVersion,
        identifier: String,
        title: String,
        summary: String,
        runtime: Runtime,
        simulation: Simulation,
        presentation: Presentation
    ) {
        self.schemaVersion = schemaVersion
        self.identifier = identifier
        self.title = title
        self.summary = summary
        self.runtime = runtime
        self.simulation = simulation
        self.presentation = presentation
    }

    public struct Runtime: Codable, Sendable, Equatable {
        public enum Kind: String, Codable, Sendable {
            /// Optimizer-free AVBD vector PPO deployment bundle.
            case vectorPPO = "avbd-vector-ppo-v1"
            /// Converted Unitree RL Gym recurrent policy.
            case unitreeH1Recurrent = "unitree-h1-recurrent-v1"
        }

        public var kind: Kind
        /// Semantic role -> bundle-relative file path.
        public var files: [String: String]

        public init(kind: Kind, files: [String: String]) {
            self.kind = kind
            self.files = files
        }
    }

    public struct Simulation: Codable, Sendable, Equatable {
        public var task: String
        public var taskRevision: Int
        public var seed: UInt64
        public var maxEpisodeSteps: Int
        public var simulationStepSeconds: Float
        public var controlDecimation: Int
        public var includeInteractiveRobustnessProbes: Bool
        public var options: [String: Float]

        public init(
            task: String,
            taskRevision: Int,
            seed: UInt64,
            maxEpisodeSteps: Int,
            simulationStepSeconds: Float,
            controlDecimation: Int,
            includeInteractiveRobustnessProbes: Bool = false,
            options: [String: Float]
        ) {
            self.task = task
            self.taskRevision = taskRevision
            self.seed = seed
            self.maxEpisodeSteps = maxEpisodeSteps
            self.simulationStepSeconds = simulationStepSeconds
            self.controlDecimation = controlDecimation
            self.includeInteractiveRobustnessProbes =
                includeInteractiveRobustnessProbes
            self.options = options
        }
    }

    public struct Presentation: Codable, Sendable, Equatable {
        public var cameraPresets: [CameraPreset]
        public var controls: [Control]
        public var metrics: [Metric]

        public init(
            cameraPresets: [CameraPreset],
            controls: [Control] = [],
            metrics: [Metric] = []
        ) {
            self.cameraPresets = cameraPresets
            self.controls = controls
            self.metrics = metrics
        }
    }

    public struct CameraPreset: Codable, Sendable, Equatable, Identifiable {
        public var id: String
        public var label: String
        /// Named task/session anchor. `world` uses `target` as an absolute
        /// point; other anchors add `target` as an anchor-relative vector.
        public var anchor: String
        public var target: [Float]
        public var offset: [Float]
        public var distance: Float
        public var azimuth: Float
        public var elevation: Float

        public init(
            id: String,
            label: String,
            anchor: String = "robot",
            target: [Float] = [0, 0, 0],
            offset: [Float] = [0, 0, 0],
            distance: Float,
            azimuth: Float,
            elevation: Float
        ) {
            self.id = id
            self.label = label
            self.anchor = anchor
            self.target = target
            self.offset = offset
            self.distance = distance
            self.azimuth = azimuth
            self.elevation = elevation
        }
    }

    public struct Control: Codable, Sendable, Equatable, Identifiable {
        public enum Kind: String, Codable, Sendable {
            case slider
            case toggle
            case button
        }

        public var id: String
        public var label: String
        public var kind: Kind
        public var defaultValue: Float?
        public var minimum: Float?
        public var maximum: Float?
        public var step: Float?
        public var format: String?
        /// Named capability implemented by the selected runtime/task ABI.
        public var command: String?
        /// Command argument name -> value-bearing slider/toggle control id.
        public var arguments: [String: String]?

        public init(
            id: String,
            label: String,
            kind: Kind,
            defaultValue: Float? = nil,
            minimum: Float? = nil,
            maximum: Float? = nil,
            step: Float? = nil,
            format: String? = nil,
            command: String? = nil,
            arguments: [String: String]? = nil
        ) {
            self.id = id
            self.label = label
            self.kind = kind
            self.defaultValue = defaultValue
            self.minimum = minimum
            self.maximum = maximum
            self.step = step
            self.format = format
            self.command = command
            self.arguments = arguments
        }
    }

    public struct Metric: Codable, Sendable, Equatable, Identifiable {
        public var id: String
        public var label: String
        /// `replay/*`, `task/*`, or `metric/*` source exposed by a session.
        public var source: String
        public var format: String
        public var unit: String?

        public init(
            id: String,
            label: String,
            source: String,
            format: String = "%.3f",
            unit: String? = nil
        ) {
            self.id = id
            self.label = label
            self.source = source
            self.format = format
            self.unit = unit
        }
    }
}

public enum PolicyBundleError: Error, LocalizedError, Equatable {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case let .invalid(message): "invalid policy bundle: \(message)"
        }
    }
}

public struct LoadedPolicyBundle: Sendable {
    public let directory: URL
    public let manifest: PolicyBundleManifest
    public let manifestData: Data
    public let manifestSHA256: String
    /// Identity of the manifest plus every declared runtime file. This is a
    /// content address for import/deduplication, not a qualification claim.
    public let contentSHA256: String

    public func fileURL(role: String) throws -> URL {
        guard let relative = manifest.runtime.files[role] else {
            throw PolicyBundleError.invalid(
                "runtime file role '\(role)' is not declared")
        }
        return try PolicyBundleLoader.resolve(
            relativePath: relative, beneath: directory)
    }
}

public enum PolicyBundleLoader {
    public static func load(directory: URL) throws -> LoadedPolicyBundle {
        let root = directory.standardizedFileURL
        let rootValues = try root.resourceValues(
            forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isDirectory == true,
              rootValues.isSymbolicLink != true else {
            throw PolicyBundleError.invalid(
                "bundle root must be a real directory: \(root.path)")
        }
        let manifestURL = root.appendingPathComponent(
            PolicyBundleManifest.fileName, isDirectory: false)
        // Inspect the complete tree before opening any attacker-controlled
        // path. In particular, policy-bundle.json itself must not be a
        // symlink to a file, device, or stream outside the selected bundle.
        try validateTree(root)
        let data = try Data(contentsOf: manifestURL)
        try validateManifestJSONShape(data)
        let manifest: PolicyBundleManifest
        do {
            manifest = try JSONDecoder().decode(
                PolicyBundleManifest.self, from: data)
        } catch {
            throw PolicyBundleError.invalid(
                "cannot decode \(PolicyBundleManifest.fileName): \(error)")
        }
        try validate(manifest, root: root)
        try validateRuntimeContract(manifest, root: root)
        return LoadedPolicyBundle(
            directory: root,
            manifest: manifest,
            manifestData: data,
            manifestSHA256: sha256(data),
            contentSHA256: try contentSHA256(
                manifest: manifest, manifestData: data, root: root))
    }

    public static func discover(beneath root: URL) -> [LoadedPolicyBundle] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            return []
        }
        var bundles = [LoadedPolicyBundle]()
        for case let url as URL in enumerator
            where url.lastPathComponent == PolicyBundleManifest.fileName {
            enumerator.skipDescendants()
            if let bundle = try? load(
                directory: url.deletingLastPathComponent()) {
                bundles.append(bundle)
            }
        }
        return bundles.sorted {
            if $0.manifest.title == $1.manifest.title {
                return $0.manifest.identifier < $1.manifest.identifier
            }
            return $0.manifest.title < $1.manifest.title
        }
    }

    static func resolve(relativePath: String, beneath root: URL) throws -> URL {
        let components = relativePath.split(
            separator: "/", omittingEmptySubsequences: false)
        guard !relativePath.isEmpty, !relativePath.hasPrefix("/"),
              components.allSatisfy({
                  !$0.isEmpty && $0 != "." && $0 != ".."
              }) else {
            throw PolicyBundleError.invalid(
                "runtime path must be nonempty and bundle-relative: \(relativePath)")
        }
        let resolved = root.appendingPathComponent(relativePath)
            .standardizedFileURL
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard resolved.path.hasPrefix(prefix) else {
            throw PolicyBundleError.invalid(
                "runtime path escapes the bundle: \(relativePath)")
        }
        let values = try resolved.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true,
              values.isRegularFile == true else {
            throw PolicyBundleError.invalid(
                "runtime path is not a regular in-bundle file: \(relativePath)")
        }
        return resolved
    }

    private static func validate(
        _ manifest: PolicyBundleManifest, root: URL
    ) throws {
        guard manifest.schemaVersion
                == PolicyBundleManifest.currentSchemaVersion else {
            throw PolicyBundleError.invalid(
                "unsupported schema version \(manifest.schemaVersion)")
        }
        try validateIdentifier(manifest.identifier, field: "identifier")
        guard !manifest.title.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty,
              !manifest.summary.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty else {
            throw PolicyBundleError.invalid("title and summary are required")
        }
        let simulation = manifest.simulation
        guard !simulation.task.isEmpty,
              simulation.taskRevision > 0,
              simulation.maxEpisodeSteps > 0,
              simulation.simulationStepSeconds.isFinite,
              simulation.simulationStepSeconds > 0,
              simulation.controlDecimation > 0,
              simulation.options.values.allSatisfy(\.isFinite) else {
            throw PolicyBundleError.invalid(
                "simulation task, timing, or options are invalid")
        }
        guard !manifest.runtime.files.isEmpty else {
            throw PolicyBundleError.invalid("runtime files are empty")
        }
        guard Set(manifest.runtime.files.values).count
                == manifest.runtime.files.count else {
            throw PolicyBundleError.invalid(
                "runtime file roles must reference distinct files")
        }
        for role in manifest.runtime.files.keys.sorted() {
            try validateIdentifier(role, field: "runtime file role")
            _ = try resolve(
                relativePath: manifest.runtime.files[role]!, beneath: root)
        }

        let cameras = manifest.presentation.cameraPresets
        guard !cameras.isEmpty else {
            throw PolicyBundleError.invalid(
                "at least one camera preset is required")
        }
        try validateUnique(cameras.map(\.id), kind: "camera")
        try validateUnique(
            manifest.presentation.controls.map(\.id), kind: "control")
        try validateUnique(
            manifest.presentation.metrics.map(\.id), kind: "metric")
        let controlIDs = Set(manifest.presentation.controls.map(\.id))
        for camera in cameras {
            try validateIdentifier(camera.id, field: "camera id")
            guard !camera.label.trimmingCharacters(
                    in: .whitespacesAndNewlines).isEmpty,
                  camera.target.count == 3, camera.offset.count == 3,
                  (camera.target + camera.offset).allSatisfy(\.isFinite),
                  camera.distance.isFinite, camera.distance > 0,
                  camera.azimuth.isFinite, camera.elevation.isFinite else {
                throw PolicyBundleError.invalid(
                    "camera '\(camera.id)' has invalid geometry")
            }
        }
        for control in manifest.presentation.controls {
            try validateIdentifier(control.id, field: "control id")
            guard !control.label.trimmingCharacters(
                    in: .whitespacesAndNewlines).isEmpty else {
                throw PolicyBundleError.invalid(
                    "control '\(control.id)' has an empty label")
            }
            switch control.kind {
            case .slider:
                let validStep = control.step.map { step in
                    step.isFinite && step > 0
                        && step <= (control.maximum ?? 0)
                            - (control.minimum ?? 0)
                } ?? true
                guard let value = control.defaultValue,
                      let minimum = control.minimum,
                      let maximum = control.maximum,
                      value.isFinite, minimum.isFinite, maximum.isFinite,
                      minimum < maximum, (minimum...maximum).contains(value),
                      validStep,
                      control.command == nil else {
                    throw PolicyBundleError.invalid(
                        "slider '\(control.id)' has an invalid range or command")
                }
            case .toggle:
                guard let value = control.defaultValue,
                      value == 0 || value == 1,
                      control.command == nil else {
                    throw PolicyBundleError.invalid(
                        "toggle '\(control.id)' must default to 0 or 1")
                }
            case .button:
                guard let command = control.command, !command.isEmpty,
                      control.defaultValue == nil,
                      control.minimum == nil, control.maximum == nil else {
                    throw PolicyBundleError.invalid(
                        "button '\(control.id)' must declare only a command")
                }
                for reference in control.arguments.map({ Array($0.values) }) ?? []
                    where !controlIDs.contains(reference) {
                    throw PolicyBundleError.invalid(
                        "button '\(control.id)' references unknown control "
                            + "'\(reference)'")
                }
            }
        }
        for metric in manifest.presentation.metrics {
            try validateIdentifier(metric.id, field: "metric id")
            guard !metric.label.isEmpty, !metric.source.isEmpty,
                  !metric.format.isEmpty else {
                throw PolicyBundleError.invalid(
                    "metric '\(metric.id)' is incomplete")
            }
        }
    }

    private static func validateRuntimeContract(
        _ manifest: PolicyBundleManifest, root: URL
    ) throws {
        switch manifest.runtime.kind {
        case .vectorPPO:
            let required = [
                "metadata": VectorPolicyDeploymentBundle.metadataFileName,
                "policy": VectorPolicyDeploymentBundle.policyFileName,
                "trainingState":
                    VectorPolicyDeploymentBundle.trainingStateFileName,
                "deploymentManifest":
                    VectorPolicyDeploymentBundle.manifestFileName,
            ]
            for (role, canonicalName) in required {
                guard manifest.runtime.files[role] == canonicalName else {
                    throw PolicyBundleError.invalid(
                        "vector PPO runtime role '\(role)' must use "
                            + "canonical file '\(canonicalName)'")
                }
            }
            let metadataURL = try resolve(
                relativePath: manifest.runtime.files["metadata"]!, beneath: root)
            let deploymentURL = try resolve(
                relativePath: manifest.runtime.files["deploymentManifest"]!,
                beneath: root)
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self, from: Data(contentsOf: metadataURL))
            let deployment = try JSONDecoder().decode(
                VectorPolicyDeploymentManifest.self,
                from: Data(contentsOf: deploymentURL))
            let simulation = manifest.simulation
            guard metadata.task == simulation.task,
                  metadata.taskRevision == simulation.taskRevision,
                  metadata.maxEpisodeSteps == simulation.maxEpisodeSteps,
                  metadata.simulationStep == simulation.simulationStepSeconds,
                  metadata.controlDecimation == simulation.controlDecimation,
                  metadata.taskConfiguration == simulation.options,
                  deployment.task == simulation.task,
                  deployment.taskRevision == simulation.taskRevision,
                  deployment.simulationStepSeconds
                    == simulation.simulationStepSeconds,
                  deployment.controlDecimation
                    == simulation.controlDecimation,
                  deployment.taskConfiguration == simulation.options else {
                throw PolicyBundleError.invalid(
                    "simulation manifest disagrees with vector PPO checkpoint")
            }
        case .unitreeH1Recurrent:
            for (role, canonicalName) in [
                "manifest": "manifest.json",
                "policy": "policy.safetensors",
                "license": "LICENSE",
            ] {
                guard manifest.runtime.files[role] == canonicalName else {
                    throw PolicyBundleError.invalid(
                        "Unitree recurrent runtime role '\(role)' must use "
                            + "canonical file '\(canonicalName)'")
                }
            }
            let policyManifestURL = try resolve(
                relativePath: manifest.runtime.files["manifest"]!,
                beneath: root)
            let policyManifest = try JSONDecoder().decode(
                UnitreeH1PolicyManifest.self,
                from: Data(contentsOf: policyManifestURL))
            let simulation = manifest.simulation
            let expectedOptions: Set<String> = [
                "commandX", "commandY", "commandYaw", "solverIterations",
            ]
            let solverIterations = simulation.options["solverIterations"]
                .flatMap(Int.init(exactly:))
            guard policyManifest.format == "avbd-unitree-h1-lstm-v1",
                  simulation.task == "unitree-h1-sim2sim-v0",
                  simulation.taskRevision == 1,
                  // This ABI currently owns an interactive projectile. Do
                  // not accept a supposedly exact manifest that disables it
                  // while reconstructing a different scene at runtime.
                  simulation.includeInteractiveRobustnessProbes,
                  Set(simulation.options.keys) == expectedOptions,
                  solverIterations.map({ $0 > 0 }) == true,
                  policyManifest.weightsFile
                    == manifest.runtime.files["policy"],
                  policyManifest.source.licenseFile
                    == manifest.runtime.files["license"],
                  policyManifest.control.physicsTimeStep
                    == simulation.simulationStepSeconds,
                  policyManifest.control.controlDecimation
                    == simulation.controlDecimation,
                  policyManifest.control.defaultCommand == [
                    simulation.options["commandX"] ?? 0,
                    simulation.options["commandY"] ?? 0,
                    simulation.options["commandYaw"] ?? 0,
                  ] else {
                throw PolicyBundleError.invalid(
                    "simulation manifest disagrees with Unitree policy manifest")
            }
        }
    }

    private static func validateTree(_ root: URL) throws {
        let keys: Set<URLResourceKey> = [
            .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
        ]
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: Array(keys),
            options: []) else {
            throw PolicyBundleError.invalid("cannot enumerate bundle files")
        }
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: keys)
            guard values.isSymbolicLink != true,
                  values.isDirectory == true
                    || values.isRegularFile == true else {
                throw PolicyBundleError.invalid(
                    "bundle contains a symlink or special file: "
                        + url.lastPathComponent)
            }
        }
    }

    private static func validateManifestJSONShape(_ data: Data) throws {
        let raw: Any
        do {
            raw = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw PolicyBundleError.invalid(
                "cannot parse \(PolicyBundleManifest.fileName): \(error)")
        }
        let root = try object(raw, label: "manifest")
        try exactKeys(root, [
            "schemaVersion", "identifier", "title", "summary", "runtime",
            "simulation", "presentation",
        ], label: "manifest")
        let runtime = try object(root["runtime"], label: "runtime")
        try exactKeys(runtime, ["kind", "files"], label: "runtime")
        let simulation = try object(root["simulation"], label: "simulation")
        try exactKeys(simulation, [
            "task", "taskRevision", "seed", "maxEpisodeSteps",
            "simulationStepSeconds", "controlDecimation",
            "includeInteractiveRobustnessProbes", "options",
        ], label: "simulation")
        let presentation = try object(
            root["presentation"], label: "presentation")
        try exactKeys(presentation, [
            "cameraPresets", "controls", "metrics",
        ], label: "presentation")
        for (index, value) in try array(
            presentation["cameraPresets"], label: "cameraPresets").enumerated() {
            try exactKeys(try object(value, label: "cameraPresets[\(index)]"), [
                "id", "label", "anchor", "target", "offset", "distance",
                "azimuth", "elevation",
            ], label: "cameraPresets[\(index)]")
        }
        let controlKeys: Set<String> = [
            "id", "label", "kind", "defaultValue", "minimum", "maximum",
            "step", "format", "command", "arguments",
        ]
        for (index, value) in try array(
            presentation["controls"], label: "controls").enumerated() {
            let item = try object(value, label: "controls[\(index)]")
            try allowedKeys(item, controlKeys, required: [
                "id", "label", "kind",
            ], label: "controls[\(index)]")
        }
        let metricKeys: Set<String> = [
            "id", "label", "source", "format", "unit",
        ]
        for (index, value) in try array(
            presentation["metrics"], label: "metrics").enumerated() {
            try allowedKeys(
                try object(value, label: "metrics[\(index)]"), metricKeys,
                required: ["id", "label", "source", "format"],
                label: "metrics[\(index)]")
        }
    }

    private static func object(_ value: Any?, label: String) throws
        -> [String: Any]
    {
        guard let value = value as? [String: Any] else {
            throw PolicyBundleError.invalid("\(label) must be an object")
        }
        return value
    }

    private static func array(_ value: Any?, label: String) throws -> [Any] {
        guard let value = value as? [Any] else {
            throw PolicyBundleError.invalid("\(label) must be an array")
        }
        return value
    }

    private static func exactKeys(
        _ object: [String: Any], _ expected: Set<String>, label: String
    ) throws {
        guard Set(object.keys) == expected else {
            throw PolicyBundleError.invalid(
                "\(label) has missing or unknown fields")
        }
    }

    private static func allowedKeys(
        _ object: [String: Any], _ allowed: Set<String>,
        required: Set<String>, label: String
    ) throws {
        let keys = Set(object.keys)
        guard required.isSubset(of: keys), keys.isSubset(of: allowed) else {
            throw PolicyBundleError.invalid(
                "\(label) has missing or unknown fields")
        }
    }

    private static func validateIdentifier(
        _ value: String, field: String
    ) throws {
        guard !value.isEmpty,
              value.unicodeScalars.allSatisfy({ scalar in
                  CharacterSet.alphanumerics.contains(scalar)
                    || scalar == "-" || scalar == "_" || scalar == "."
              }) else {
            throw PolicyBundleError.invalid("\(field) '\(value)' is invalid")
        }
    }

    private static func validateUnique(
        _ values: [String], kind: String
    ) throws {
        guard Set(values).count == values.count else {
            throw PolicyBundleError.invalid("duplicate \(kind) id")
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func contentSHA256(
        manifest: PolicyBundleManifest, manifestData: Data, root: URL
    ) throws -> String {
        var inventory = ["manifest\0\(sha256(manifestData))\n"]
        for role in manifest.runtime.files.keys.sorted() {
            let relative = manifest.runtime.files[role]!
            let url = try resolve(relativePath: relative, beneath: root)
            let digest = sha256(try Data(contentsOf: url))
            inventory.append("\(role)\0\(relative)\0\(digest)\n")
        }
        return sha256(Data(inventory.joined().utf8))
    }
}

/// Independent identity for the converted Unitree release. These values are
/// supplied only by the app's release index; the candidate manifest cannot
/// certify its own source, conversion, license, or golden parity record.
public struct UnitreeH1ReleaseIdentity: Codable, Sendable, Equatable {
    public var manifestSHA256: String
    public var sourceRevision: String
    public var sourceCheckpointSHA256: String
    public var weightsSHA256: String
    public var licenseSHA256: String
    public var goldenSequenceSHA256: String

    public init(
        manifestSHA256: String,
        sourceRevision: String,
        sourceCheckpointSHA256: String,
        weightsSHA256: String,
        licenseSHA256: String,
        goldenSequenceSHA256: String
    ) {
        self.manifestSHA256 = manifestSHA256
        self.sourceRevision = sourceRevision
        self.sourceCheckpointSHA256 = sourceCheckpointSHA256
        self.weightsSHA256 = weightsSHA256
        self.licenseSHA256 = licenseSHA256
        self.goldenSequenceSHA256 = goldenSequenceSHA256
    }
}

/// Qualification is external to a bundle: an imported bundle cannot certify
/// itself. A release index shipped with the app authenticates exact bundle
/// manifest bytes and, for learned runtimes, the commissioned artifact.
public struct PolicyBundleReleaseIndex: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var releases: [Release]

    public init(schemaVersion: Int = 1, releases: [Release]) {
        self.schemaVersion = schemaVersion
        self.releases = releases
    }

    public struct Release: Codable, Sendable, Equatable {
        public enum Qualification: String, Codable, Sendable {
            case accepted
            case externalParityVerified
        }

        public var bundleIdentifier: String
        /// Relative to the packaged `checkpoints` directory.
        public var bundleRelativeDirectory: String
        public var bundleManifestSHA256: String
        public var qualification: Qualification
        /// Repository-relative evidence or immutable external source manifest.
        public var evidenceRelativePath: String
        public var acceptanceAggregateRelativePath: String?
        public var deploymentManifestRelativePath: String?
        public var expectedTaskRevision: Int
        public var expectedCheckpointFingerprint: String?
        public var expectedDeploymentManifestSHA256: String?
        public var unitreeH1ReleaseIdentity: UnitreeH1ReleaseIdentity?

        public init(
            bundleIdentifier: String,
            bundleRelativeDirectory: String,
            bundleManifestSHA256: String,
            qualification: Qualification,
            evidenceRelativePath: String,
            acceptanceAggregateRelativePath: String? = nil,
            deploymentManifestRelativePath: String? = nil,
            expectedTaskRevision: Int,
            expectedCheckpointFingerprint: String? = nil,
            expectedDeploymentManifestSHA256: String? = nil,
            unitreeH1ReleaseIdentity: UnitreeH1ReleaseIdentity? = nil
        ) {
            self.bundleIdentifier = bundleIdentifier
            self.bundleRelativeDirectory = bundleRelativeDirectory
            self.bundleManifestSHA256 = bundleManifestSHA256
            self.qualification = qualification
            self.evidenceRelativePath = evidenceRelativePath
            self.acceptanceAggregateRelativePath =
                acceptanceAggregateRelativePath
            self.deploymentManifestRelativePath =
                deploymentManifestRelativePath
            self.expectedTaskRevision = expectedTaskRevision
            self.expectedCheckpointFingerprint = expectedCheckpointFingerprint
            self.expectedDeploymentManifestSHA256 =
                expectedDeploymentManifestSHA256
            self.unitreeH1ReleaseIdentity = unitreeH1ReleaseIdentity
        }
    }

    public func release(for bundle: LoadedPolicyBundle) -> Release? {
        releases.first {
            $0.bundleIdentifier == bundle.manifest.identifier
                && $0.bundleManifestSHA256 == bundle.manifestSHA256
                && $0.expectedTaskRevision
                    == bundle.manifest.simulation.taskRevision
                && (($0.qualification == .accepted
                        && bundle.manifest.runtime.kind == .vectorPPO
                        && $0.expectedCheckpointFingerprint != nil
                        && $0.expectedDeploymentManifestSHA256 != nil
                        && $0.unitreeH1ReleaseIdentity == nil)
                    || ($0.qualification == .externalParityVerified
                        && bundle.manifest.runtime.kind
                            == .unitreeH1Recurrent
                        && $0.expectedCheckpointFingerprint == nil
                        && $0.expectedDeploymentManifestSHA256 == nil
                        && $0.unitreeH1ReleaseIdentity != nil))
        }
    }

    public static func load(from url: URL) throws
        -> PolicyBundleReleaseIndex
    {
        let data = try Data(contentsOf: url)
        try validateJSONShape(data)
        let index = try JSONDecoder().decode(
            PolicyBundleReleaseIndex.self, from: data)
        guard index.schemaVersion == 1,
              Set(index.releases.map(\.bundleIdentifier)).count
                == index.releases.count,
              Set(index.releases.map(\.bundleRelativeDirectory)).count
                == index.releases.count else {
            throw PolicyBundleError.invalid(
                "release index schema, identifiers, or directories are invalid")
        }
        for release in index.releases {
            guard !release.bundleIdentifier.isEmpty,
                  release.bundleIdentifier.unicodeScalars.allSatisfy({ scalar in
                      CharacterSet.alphanumerics.contains(scalar)
                        || scalar == "-" || scalar == "_" || scalar == "."
                  }),
                  release.expectedTaskRevision > 0,
                  validRelativePath(release.bundleRelativeDirectory),
                  validRelativePath(release.evidenceRelativePath),
                  release.acceptanceAggregateRelativePath.map(
                    validRelativePath) ?? true,
                  release.deploymentManifestRelativePath.map(
                    validRelativePath) ?? true,
                  isSHA256(release.bundleManifestSHA256),
                  release.expectedCheckpointFingerprint.map(isSHA256) ?? true,
                  release.expectedDeploymentManifestSHA256.map(isSHA256)
                    ?? true else {
                throw PolicyBundleError.invalid(
                    "release '\(release.bundleIdentifier)' has an invalid identity")
            }
            switch release.qualification {
            case .accepted:
                guard release.acceptanceAggregateRelativePath != nil,
                      release.deploymentManifestRelativePath != nil,
                      release.expectedCheckpointFingerprint != nil,
                      release.expectedDeploymentManifestSHA256 != nil,
                      release.unitreeH1ReleaseIdentity == nil else {
                    throw PolicyBundleError.invalid(
                        "accepted release '\(release.bundleIdentifier)' is incomplete")
                }
            case .externalParityVerified:
                guard release.acceptanceAggregateRelativePath == nil,
                      release.deploymentManifestRelativePath == nil,
                      release.expectedCheckpointFingerprint == nil,
                      release.expectedDeploymentManifestSHA256 == nil,
                      let identity = release.unitreeH1ReleaseIdentity,
                      isSHA256(identity.manifestSHA256),
                      isSHA256(identity.sourceCheckpointSHA256),
                      isSHA256(identity.weightsSHA256),
                      isSHA256(identity.licenseSHA256),
                      isSHA256(identity.goldenSequenceSHA256),
                      identity.sourceRevision.count == 40,
                      identity.sourceRevision
                        == identity.sourceRevision.lowercased(),
                      identity.sourceRevision.allSatisfy(\.isHexDigit) else {
                    throw PolicyBundleError.invalid(
                        "external release '\(release.bundleIdentifier)' is incomplete")
                }
            }
        }
        return index
    }

    private static func validateJSONShape(_ data: Data) throws {
        let raw: Any
        do {
            raw = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw PolicyBundleError.invalid(
                "cannot parse release index: \(error)")
        }
        guard let root = raw as? [String: Any],
              Set(root.keys) == ["schemaVersion", "releases"],
              let releases = root["releases"] as? [Any] else {
            throw PolicyBundleError.invalid(
                "release index has missing or unknown fields")
        }
        let allowed: Set<String> = [
            "bundleIdentifier", "bundleRelativeDirectory",
            "bundleManifestSHA256", "qualification",
            "evidenceRelativePath", "acceptanceAggregateRelativePath",
            "deploymentManifestRelativePath", "expectedTaskRevision",
            "expectedCheckpointFingerprint",
            "expectedDeploymentManifestSHA256", "unitreeH1ReleaseIdentity",
        ]
        let required: Set<String> = [
            "bundleIdentifier", "bundleRelativeDirectory",
            "bundleManifestSHA256", "qualification",
            "evidenceRelativePath", "expectedTaskRevision",
        ]
        let identityKeys: Set<String> = [
            "manifestSHA256", "sourceRevision", "sourceCheckpointSHA256",
            "weightsSHA256", "licenseSHA256", "goldenSequenceSHA256",
        ]
        for (index, value) in releases.enumerated() {
            guard let release = value as? [String: Any] else {
                throw PolicyBundleError.invalid(
                    "release[\(index)] must be an object")
            }
            let keys = Set(release.keys)
            guard required.isSubset(of: keys), keys.isSubset(of: allowed) else {
                throw PolicyBundleError.invalid(
                    "release[\(index)] has missing or unknown fields")
            }
            let expectedKeys: Set<String>
            switch release["qualification"] as? String {
            case "accepted":
                expectedKeys = required.union([
                    "acceptanceAggregateRelativePath",
                    "deploymentManifestRelativePath",
                    "expectedCheckpointFingerprint",
                    "expectedDeploymentManifestSHA256",
                ])
            case "externalParityVerified":
                expectedKeys = required.union(["unitreeH1ReleaseIdentity"])
            default:
                throw PolicyBundleError.invalid(
                    "release[\(index)] has an invalid qualification")
            }
            guard keys == expectedKeys else {
                throw PolicyBundleError.invalid(
                    "release[\(index)] has fields for the wrong qualification")
            }
            if let identity = release["unitreeH1ReleaseIdentity"] {
                guard let identity = identity as? [String: Any],
                      Set(identity.keys) == identityKeys else {
                    throw PolicyBundleError.invalid(
                        "release[\(index)] has an invalid external identity")
                }
            }
        }
    }

    private static func validRelativePath(_ value: String) -> Bool {
        guard !value.isEmpty, !value.hasPrefix("/") else { return false }
        return value.split(separator: "/", omittingEmptySubsequences: false)
            .allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value == value.lowercased()
            && value.allSatisfy { $0.isHexDigit }
    }
}
