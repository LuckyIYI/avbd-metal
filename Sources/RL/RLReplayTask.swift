import Foundation
import SimCore
import PhysicsAVBD

/// Task-owned presentation/capability boundary for data-driven replay.
///
/// The app and policy bundle never downcast a task. A task opts into replay by
/// exposing named anchors, scalar values, and commands. Those names form the
/// stable task ABI referenced by `policy-bundle.json`; adding a new policy for
/// an existing task requires no Swift UI code.
public protocol RLReplayTask: VectorizedRLTask {
    var replaySolver: GPUSolver { get }
    var replayCapabilities: RLReplayCapabilities { get }
    func replayAnchor(named name: String, environment: Int) -> F3?
    func replayValues(
        environment: Int, latestStep: RLStepBatch?
    ) -> [String: Float]
    func performReplayCommand(
        _ command: String,
        arguments: [String: Float],
        environment: Int
    ) throws -> RLReplayCommandEffect
}

public struct RLReplayCapabilities: Sendable, Equatable {
    public var anchors: Set<String>
    public var values: Set<String>
    public var commands: Set<String>

    public init(
        anchors: Set<String>, values: Set<String>, commands: Set<String>
    ) {
        self.anchors = anchors
        self.values = values
        self.commands = commands
    }
}

public enum RLReplayCommandEffect: Sendable, Equatable {
    case none
    case reset
}

public extension RLReplayTask {
    func performReplayCommand(
        _ command: String,
        arguments: [String: Float],
        environment: Int
    ) throws -> RLReplayCommandEffect {
        throw RLEnvironmentError.invalidConfiguration(
            "task \(spec.id) does not support replay command '\(command)'")
    }

    func checkedReplayEnvironment(_ environment: Int) throws {
        guard (0..<spec.numEnvironments).contains(environment) else {
            throw RLEnvironmentError.invalidEnvironmentIndex(environment)
        }
    }
}

extension HumanoidIsaacVelocityTask: RLReplayTask {
    public var replaySolver: GPUSolver { environment.solver }

    public var replayCapabilities: RLReplayCapabilities {
        var anchors: Set<String> = ["robot", "course", "world"]
        var values: Set<String> = [
            "task/root-x", "task/root-y", "task/root-height",
            "task/planar-speed", "task/yaw-rate", "task/command-forward",
            "task/command-lateral", "task/command-yaw",
        ]
        var commands = Set<String>()
        if usesPointGoal {
            anchors.insert("goal")
            values.insert("task/goal-distance")
            commands.formUnion(["set-goal", "randomize-goal"])
        }
        if hasProjectile(environment: 0) {
            commands.insert("throw-projectile")
        }
        return .init(anchors: anchors, values: values, commands: commands)
    }

    public func replayAnchor(
        named name: String, environment index: Int
    ) -> F3? {
        guard (0..<spec.numEnvironments).contains(index) else { return nil }
        switch name {
        case "robot": return environment.states()[index].root.position
        case "course": return currentCommandProjection(environment: index)
        case "goal":
            return usesPointGoal ? currentGoalPosition(environment: index) : nil
        case "world": return .zero
        default: return nil
        }
    }

    public func replayValues(
        environment index: Int, latestStep: RLStepBatch?
    ) -> [String: Float] {
        guard (0..<spec.numEnvironments).contains(index) else { return [:] }
        let state = environment.states()[index]
        let command = currentCommand(environment: index)
        let planarSpeed = hypot(
            state.root.linearVelocity.x, state.root.linearVelocity.y)
        var values: [String: Float] = [
            "task/root-x": state.root.position.x,
            "task/root-y": state.root.position.y,
            "task/root-height": state.root.position.z,
            "task/planar-speed": planarSpeed,
            "task/yaw-rate": state.root.angularVelocity.z,
            "task/command-forward": command.x,
            "task/command-lateral": command.y,
            "task/command-yaw": command.z,
        ]
        if usesPointGoal {
            let goal = currentGoalPosition(environment: index)
            let delta = goal - state.root.position
            values["task/goal-distance"] = hypot(delta.x, delta.y)
        }
        if let latestStep {
            for (name, rows) in latestStep.metrics where rows.indices.contains(index) {
                values["metric/\(name)"] = rows[index]
            }
        }
        return values
    }

    public func performReplayCommand(
        _ command: String,
        arguments: [String: Float],
        environment index: Int
    ) throws -> RLReplayCommandEffect {
        try checkedReplayEnvironment(index)
        switch command {
        case "set-goal":
            guard usesPointGoal,
                  let bearing = arguments["bearingDegrees"],
                  let distance = arguments["distance"],
                  bearing.isFinite, distance.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "set-goal requires finite bearingDegrees and distance")
            }
            let radians = bearing * .pi / 180
            try setGoal(
                environment: index,
                direction: F3(cos(radians), sin(radians), 0),
                distance: distance)
            return .reset
        case "randomize-goal":
            guard usesPointGoal else {
                throw RLEnvironmentError.invalidConfiguration(
                    "randomize-goal requires a point-goal task")
            }
            clearGoalOverride(environment: index)
            return .reset
        case "throw-projectile":
            guard hasProjectile(environment: index) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task scene has no interactive projectile")
            }
            let side = arguments["side"] ?? 1
            let speed = arguments["speed"] ?? 6
            throwRobustnessBoxes(
                environmentIDs: [index], sideSigns: [side], speed: speed)
            return .none
        default:
            throw RLEnvironmentError.invalidConfiguration(
                "task \(spec.id) does not support replay command '\(command)'")
        }
    }
}

extension Arachne15LocomotionTask: RLReplayTask {
    public var replaySolver: GPUSolver { environment.solver }

    public var replayCapabilities: RLReplayCapabilities {
        var anchors: Set<String> = ["robot", "world"]
        var values: Set<String> = [
            "task/root-x", "task/root-y", "task/root-height",
            "task/planar-speed", "task/yaw-rate", "task/command-forward",
            "task/command-lateral", "task/command-yaw",
        ]
        var commands = Set<String>()
        if usesPointGoal {
            anchors.formUnion(["course", "goal"])
            values.insert("task/goal-distance")
            commands.formUnion(["set-goal", "randomize-goal"])
        }
        if environment.hasProjectile(environment: 0) {
            commands.insert("throw-projectile")
        }
        return .init(anchors: anchors, values: values, commands: commands)
    }

    public func replayAnchor(
        named name: String, environment index: Int
    ) -> F3? {
        guard (0..<spec.numEnvironments).contains(index) else { return nil }
        switch name {
        case "robot": return environment.states()[index].root.position
        case "course", "goal":
            return usesPointGoal ? currentGoalPosition(environment: index) : nil
        case "world": return .zero
        default: return nil
        }
    }

    public func replayValues(
        environment index: Int, latestStep: RLStepBatch?
    ) -> [String: Float] {
        guard (0..<spec.numEnvironments).contains(index) else { return [:] }
        let state = environment.states()[index]
        let command = currentCommand(environment: index)
        var values: [String: Float] = [
            "task/root-x": state.root.position.x,
            "task/root-y": state.root.position.y,
            "task/root-height": state.root.position.z,
            "task/planar-speed": hypot(
                state.root.linearVelocity.x, state.root.linearVelocity.y),
            "task/yaw-rate": state.root.angularVelocity.z,
            "task/command-forward": command.x,
            "task/command-lateral": command.y,
            "task/command-yaw": command.z,
        ]
        if usesPointGoal {
            values["task/goal-distance"] = currentGoalDistance(
                environment: index)
        }
        if let latestStep {
            for (name, rows) in latestStep.metrics where rows.indices.contains(index) {
                values["metric/\(name)"] = rows[index]
            }
        }
        return values
    }

    public func performReplayCommand(
        _ command: String,
        arguments: [String: Float],
        environment index: Int
    ) throws -> RLReplayCommandEffect {
        try checkedReplayEnvironment(index)
        switch command {
        case "set-goal":
            guard usesPointGoal,
                  let bearing = arguments["bearingDegrees"],
                  let distance = arguments["distance"],
                  bearing.isFinite, distance.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "set-goal requires finite bearingDegrees and distance")
            }
            let radians = bearing * .pi / 180
            try setGoal(
                environment: index,
                direction: F3(cos(radians), sin(radians), 0),
                distance: distance)
            return .reset
        case "randomize-goal":
            guard usesPointGoal else {
                throw RLEnvironmentError.invalidConfiguration(
                    "randomize-goal requires a point-goal task")
            }
            clearGoalOverride(environment: index)
            return .reset
        case "throw-projectile":
            guard environment.hasProjectile(environment: index) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task scene has no interactive projectile")
            }
            let side = arguments["side"] ?? 1
            let speed = arguments["speed"] ?? 2.5
            environment.throwBoxes(
                environmentIDs: [index], sideSigns: [side],
                launchDistance: 0.35, speed: speed)
            return .none
        default:
            throw RLEnvironmentError.invalidConfiguration(
                "task \(spec.id) does not support replay command '\(command)'")
        }
    }
}
