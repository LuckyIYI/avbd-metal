import SimCore
import simd

/// Backend-independent task state. The viewer and headless regression use the
/// same command clock, observations and stop rule.
public struct EthernetInsertionRun {
    public let task: EthernetInsertionTask
    public private(set) var step = 0
    public private(set) var stopped = false
    public private(set) var force = F3.zero
    public private(set) var peakForce: Float = 0
    public private(set) var noseCenter = F3.zero
    public var time: Float { min(Float(step) * task.scene.settings.dt, EthernetInsertionTask.duration) }
    public var command: EthernetInsertionTask.Command {
        var c = task.command(at: time)
        if stopped {
            c.linearVelocity = .zero
            c.angularVelocity = .zero
        }
        return c
    }
    public var seated: Bool {
        !stopped && time >= EthernetInsertionTask.duration
            && noseCenter.x > EthernetInsertionTask.seatDepth - 0.0003
            && abs(noseCenter.y) < 0.0003
            && abs(noseCenter.z - EthernetInsertionTask.axisHeight) < 0.0003
    }
    public var phase: String {
        if stopped { return "Force limit — stopped" }
        if time >= EthernetInsertionTask.duration { return seated ? "Seated" : "Incomplete insertion" }
        return command.phase
    }
    public init(task: EthernetInsertionTask) { self.task = task }

    /// Observe the completed physics step before advancing the command clock.
    /// The force is the finite mount's reaction, including weight and cable
    /// loads. A stopped run continues physics with its last wrist pose held.
    public mutating func observe(toolPosition: F3, toolRotation: Quat, noseCenter: F3) {
        let c = command
        force = task.couplingForce(
            wristPosition: c.position, wristRotation: c.rotation,
            toolPosition: toolPosition, toolRotation: toolRotation)
        self.noseCenter = noseCenter
        peakForce = max(peakForce, length(force))
        if !force.x.isFinite || !force.y.isFinite || !force.z.isFinite
            || length(force) > EthernetInsertionTask.forceLimit
        {
            stopped = true
        }
        if !stopped && time < EthernetInsertionTask.duration { step += 1 }
    }
}
