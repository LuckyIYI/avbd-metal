import SimCore

extension PhysicsScene {
    /// Include one-segment cables (which have no joints), and custom cable
    /// joints authored without SceneCable metadata. Attachments do not opt
    /// their other rigid body into the cable contact policy.
    package var cableContactBodies: Set<Int> {
        var result = Set(cables.flatMap(\.bodyIDs))
        for joint in joints where joint.cable != nil {
            if joint.bodyA >= 0 { result.insert(joint.bodyA) }
            result.insert(joint.bodyB)
        }
        return result
    }
}

enum ColliderGPUFlags {
    // Collider shape flags have a separate namespace from JointGPU flags.
    static let cableContactFrame: UInt32 = 1 << 6
}
