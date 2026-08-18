import Foundation
import simd
import AVBDCore
import AVBDLearn
import MLX

// Bridges the trained LeWM + CEM planner into the Robotics Lab. MLX only
// works in xcodebuild-produced binaries (`make app-ml`); under plain SwiftPM
// the load fails gracefully and the Lab reports it.
@MainActor
enum PolicyBridge {
    /// MLX's metallib only exists in xcodebuild products; touching MLX in a
    /// plain SwiftPM binary ABORTS the process (mlx_error is not a Swift
    /// throw). Pre-flight by locating the Cmlx resource bundle.
    static var mlxAvailable: Bool {
        let exeDir = Bundle.main.bundleURL
        let candidates = [
            exeDir.appendingPathComponent("Contents/Resources/mlx-swift_Cmlx.bundle"),
            exeDir.deletingLastPathComponent().appendingPathComponent("mlx-swift_Cmlx.bundle"),
        ]
        return candidates.contains { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func install(into model: RoboticsModel) {
        guard mlxAvailable else {
            model.policyStatus = "policy: MLX needs the xcodebuild app — run `make app-ml`"
            return
        }
        // prefer the trained BC visuomotor policy (90% on Push-T);
        // fall back to the LeWM CEM planner
        do {
            if FileManager.default.fileExists(atPath: "runs/pusht/model/bc.safetensors") {
                let runner = try BCPolicyRunner(modelPath: "runs/pusht/model", latent: 192)
                model.policyAction = { env in try runner.actionChecked(env) }
                model.policyStatus = "policy: BC visuomotor (pixels -> action)"
            } else if FileManager.default.fileExists(atPath: "runs/pusht/model/lewm.safetensors") {
                let planner = try LeWMPlanner(modelPath: "runs/pusht/model")
                model.policyAction = { env in try planner.actionChecked(env) }
                model.policyStatus = "policy: LeWM loaded (CEM latent MPC)"
            } else {
                model.policyStatus = "policy: no model — train via the Training panel"
            }
        } catch {
            model.policyStatus = "policy: MLX unavailable in this build — use `make app-ml` (\(error.localizedDescription))"
        }
    }
}
