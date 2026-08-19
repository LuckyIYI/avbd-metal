import Foundation
import simd
import SimCore
import PhysicsAVBD
import Robotics
import RL
import MLXRL

// Bridges the trained LeWM + CEM planner into the Robotics Lab. MLX only
// works in xcodebuild-produced binaries (`make app-ml`); under plain SwiftPM
// the load fails gracefully and the Lab reports it.
@MainActor
enum PolicyBridge {
    private static var modelDirectoryCandidates: [URL] {
        let relativePath = "runs/pusht/model"
        let workspace = TrainingRunner.workspaceURL.appendingPathComponent(
            relativePath, isDirectory: true)
        let currentDirectory = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        ).appendingPathComponent(relativePath, isDirectory: true)
        var seen = Set<String>()
        return [workspace, currentDirectory].filter {
            seen.insert($0.standardizedFileURL.path).inserted
        }
    }

    /// MLX's metallib only exists in xcodebuild products; touching MLX in a
    /// plain SwiftPM binary ABORTS the process (mlx_error is not a Swift
    /// throw). Pre-flight by locating the compiled file, not merely a bundle
    /// directory that could be empty or incompletely packaged.
    static func requireMLXRuntime() throws {
        try MLXRuntimeResources.requireDefaultMetalLibrary()
    }

    static func install(into model: RoboticsModel) {
        // Reload is transactional: never keep driving with a stale actor when
        // its replacement is missing, incompatible, or fails validation.
        model.policyAction = nil
        do {
            try requireMLXRuntime()
        } catch {
            model.policyStatus = "policy: \(error.localizedDescription)"
            return
        }
        // Prefer every available BC candidate, then fall back to every LeWM
        // candidate. A corrupt preferred file must not mask a valid planner.
        let directories = modelDirectoryCandidates
        var failures: [String] = []
        var foundCandidate = false
        for directory in directories {
            let checkpoint = directory.appendingPathComponent("bc.safetensors")
            guard FileManager.default.fileExists(atPath: checkpoint.path) else {
                continue
            }
            foundCandidate = true
            do {
                let runner = try BCPolicyRunner(
                    modelPath: directory.path, latent: 192)
                model.policyAction = { env in try runner.actionChecked(env) }
                model.policyStatus = "policy: BC visuomotor (pixels -> action)\n"
                    + directory.path
                return
            } catch {
                failures.append("\(checkpoint.path): \(error.localizedDescription)")
            }
        }
        for directory in directories {
            let checkpoint = directory.appendingPathComponent("lewm.safetensors")
            guard FileManager.default.fileExists(atPath: checkpoint.path) else {
                continue
            }
            foundCandidate = true
            do {
                let planner = try LeWMPlanner(modelPath: directory.path)
                model.policyAction = { env in try planner.actionChecked(env) }
                model.policyStatus = "policy: LeWM loaded (CEM latent MPC, "
                    + "latent \(planner.latentDimension))\n"
                    + directory.path
                return
            } catch {
                failures.append("\(checkpoint.path): \(error.localizedDescription)")
            }
        }
        if foundCandidate {
            model.policyStatus = "policy load failed; no stale actor retained:\n"
                + failures.joined(separator: "\n")
        } else {
            model.policyStatus = "policy: no model — train via the Training panel\n"
                + TrainingRunner.workspaceURL.appendingPathComponent(
                    "runs/pusht/model", isDirectory: true).path
        }
    }
}
