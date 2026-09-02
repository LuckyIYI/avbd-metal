import Foundation
import simd
import SimCore

public enum MJCFImportError: Error, CustomStringConvertible {
    case malformedXML(String)
    case invalidAttribute(element: String, attribute: String, value: String)
    case unsupported(String)
    case missing(String)

    public var description: String {
        switch self {
        case .malformedXML(let message): return "malformed MJCF: \(message)"
        case .invalidAttribute(let element, let attribute, let value):
            return "invalid MJCF \(element).\(attribute)=\"\(value)\""
        case .unsupported(let message): return "unsupported MJCF feature: \(message)"
        case .missing(let message): return "missing MJCF data: \(message)"
        }
    }
}

public struct MJCFMotorGain: Sendable, Equatable {
    public var stiffness: Float
    public var damping: Float

    public init(stiffness: Float, damping: Float) {
        precondition(stiffness >= 0 && damping >= 0)
        self.stiffness = stiffness
        self.damping = damping
    }
}

/// Replica-local physical multipliers applied while an MJCF articulation is
/// instantiated. `mass` scales both mass and nominal inertia; `inertia`
/// applies an additional inertia-only multiplier. Keeping this at the shared
/// asset boundary lets any batched task construct a seeded population of
/// slightly different plants without adding robot-specific solver branches.
public struct MJCFDynamicsScale: Sendable, Equatable {
    public var mass: Float
    public var inertia: Float
    public var friction: Float
    public var motorTorque: Float
    public var motorStiffness: Float
    public var motorDamping: Float
    public var armature: Float

    public init(mass: Float = 1, inertia: Float = 1,
                friction: Float = 1, motorTorque: Float = 1,
                motorStiffness: Float = 1, motorDamping: Float = 1,
                armature: Float = 1) {
        let values = [mass, inertia, friction, motorTorque,
                      motorStiffness, motorDamping, armature]
        precondition(values.allSatisfy { $0.isFinite && $0 > 0 },
                     "MJCF dynamics multipliers must be finite and positive")
        self.mass = mass
        self.inertia = inertia
        self.friction = friction
        self.motorTorque = motorTorque
        self.motorStiffness = motorStiffness
        self.motorDamping = motorDamping
        self.armature = armature
    }

    public static let identity = MJCFDynamicsScale()
}

/// Complete configuration for constructing one articulation from a parsed
/// MJCF asset. Keeping these settings in a value makes task configuration
/// explicit, reusable, and safe to pass through batched scene builders.
public struct MJCFInstantiationOptions: Sendable, Equatable {
    public var worldOffset: F3
    public var defaultMotorGain: MJCFMotorGain
    public var motorGains: [String: MJCFMotorGain]
    /// Per-actuator effort-limit overrides. Keys may be either source
    /// actuator names or source joint names; actuator names take precedence.
    public var motorEffortLimits: [String: Float]
    /// Per-joint reflected armature overrides. Keys may be either source
    /// actuator names or source joint names; actuator names take precedence.
    public var jointArmatures: [String: Float]
    /// Source joint coordinates used as the reset/home configuration. The
    /// solver's zero angle is rebased to this pose.
    public var jointHomePositions: [String: Float]
    /// Weld the root body to its authored world pose.
    public var fixedBase: Bool
    /// Per-link gravity multiplier.
    public var gravityScale: Float
    /// Collision domain shared by all primitives in this instance.
    public var collisionGroup: UInt32
    public var selfCollisions: Bool
    /// Whether to preserve the legacy connected-body contact configuration.
    /// When false, the importer records every direct parent-child pair as an
    /// explicit exclusion, including nested bodies without a hinge.
    public var collideConnectedBodies: Bool
    public var inertiaFrame: MJCFInertiaFrame
    public var dynamicsScale: MJCFDynamicsScale
    /// Whether detailed render geometry should be replicated.
    public var includeVisuals: Bool

    public init(
        worldOffset: F3 = .zero,
        defaultMotorGain: MJCFMotorGain = .init(stiffness: 0, damping: 0),
        motorGains: [String: MJCFMotorGain] = [:],
        jointHomePositions: [String: Float] = [:],
        motorEffortLimits: [String: Float] = [:],
        jointArmatures: [String: Float] = [:],
        fixedBase: Bool = false,
        gravityScale: Float = 1,
        collisionGroup: UInt32 = 0,
        selfCollisions: Bool = true,
        collideConnectedBodies: Bool = true,
        inertiaFrame: MJCFInertiaFrame = .principal,
        dynamicsScale: MJCFDynamicsScale = .identity,
        includeVisuals: Bool = true
    ) {
        self.worldOffset = worldOffset
        self.defaultMotorGain = defaultMotorGain
        self.motorGains = motorGains
        precondition(motorEffortLimits.values.allSatisfy {
            $0.isFinite && $0 >= 0
        }, "MJCF motor effort limits must be finite and nonnegative")
        self.motorEffortLimits = motorEffortLimits
        precondition(jointArmatures.values.allSatisfy {
            $0.isFinite && $0 >= 0
        }, "MJCF joint armatures must be finite and nonnegative")
        self.jointArmatures = jointArmatures
        self.jointHomePositions = jointHomePositions
        self.fixedBase = fixedBase
        self.gravityScale = gravityScale
        self.collisionGroup = collisionGroup
        self.selfCollisions = selfCollisions
        self.collideConnectedBodies = collideConnectedBodies
        self.inertiaFrame = inertiaFrame
        self.dynamicsScale = dynamicsScale
        self.includeVisuals = includeVisuals
    }
}

public struct MJCFInstantiation {
    public var rootBody: Int
    public var bodiesByName: [String: Int]
    public var jointsByName: [String: Int]
    /// Joint indices in source actuator order. This is the stable policy
    /// action ordering exposed by the imported articulation.
    public var actuatorJoints: [Int]
    public var actuatorNames: [String]
    public var warnings: [String]
    /// Source link frame expressed in the corresponding solver body's
    /// inertial frame. Observation code uses this to report semantic link
    /// poses even though dynamics correctly integrate at the COM.
    public var linkFramesInBody: [String: MJCFLinkFrame]
}

public enum Arachne15CollisionProfile: String, Sendable, CaseIterable {
    case training
    case validation
}

public struct MJCFLinkFrame {
    public var position: F3
    public var rotation: Quat

    public init(position: F3, rotation: Quat) {
        self.position = position
        self.rotation = rotation
    }
}

public struct MJCFBodyPose {
    public var position: F3
    public var rotation: Quat

    public init(position: F3, rotation: Quat) {
        self.position = position
        self.rotation = rotation
    }
}

/// Orientation convention for an MJCF body's authored diagonal inertia.
///
/// MuJoCo interprets `diaginertia` in the `<inertial quat>` principal frame.
/// Some converted USD assets retain the three diagonal values and COM but do
/// not author `PhysicsMassAPI.principalAxes`; PhysX then interprets the same
/// values in the link frame. Keeping this choice explicit lets a task match a
/// particular source simulator without changing the reusable parsed asset.
public enum MJCFInertiaFrame: Sendable, Equatable {
    case principal
    case linkAligned
}

/// Parsed, reusable MJCF articulation. The parser deliberately targets the
/// rigid articulated subset used by modern locomotion assets: nested bodies,
/// principal-frame inertials, hinge joints/default classes, primitive
/// collision geoms, motors, and contact exclusions. Visual meshes remain a
/// rendering concern and never affect contact or inertia.
public struct MJCFAsset {
    public let name: String
    public let bodyNames: [String]
    public let jointNames: [String]
    public let actuatorNames: [String]
    public let warnings: [String]
    public let visualGeometryCount: Int

    fileprivate var links: [Link]
    fileprivate var actuators: [Actuator]
    fileprivate var exclusions: [(String, String)]

    /// Resolve a copied Robotics resource. Kept internal so package tests
    /// can prove that attribution payloads ship beside the assets they govern
    /// without exposing bundle layout as public API.
    private static var packagedResourceBundle: Bundle? {
        let name = "avbd-metal_Robotics.bundle"
        let roots = [Bundle.main.resourceURL, Bundle.main.bundleURL]
        for root in roots.compactMap({ $0 }) {
            if let bundle = Bundle(url: root.appendingPathComponent(name)) {
                return bundle
            }
        }
        return nil
    }

    private static var resourceBundle: Bundle {
        packagedResourceBundle ?? Bundle.module
    }

    static func bundledResourceURL(
        resource: String, withExtension fileExtension: String?,
        subdirectory: String
    ) throws -> URL {
        guard let url = resourceBundle.url(
            forResource: resource, withExtension: fileExtension,
            subdirectory: subdirectory) else {
            let suffix = fileExtension.map { ".\($0)" } ?? ""
            throw MJCFImportError.missing(
                "bundled \(subdirectory)/\(resource)\(suffix)")
        }
        return url
    }

    public static func parse(
        url: URL,
        visualNormalPolicy: SurfaceNormalPolicy = .preserveImported
    ) throws -> MJCFAsset {
        try parse(
            data: Data(contentsOf: url),
            baseURL: url.deletingLastPathComponent(),
            visualNormalPolicy: visualNormalPolicy)
    }

    /// Resolve an MJCF plus any relative visual-mesh references from the
    /// Robotics resource bundle. Physics tasks can use this generic entry
    /// point without adding importer branches for every robot.
    public static func bundled(resource: String, subdirectory: String) throws
        -> MJCFAsset {
        let url = try bundledResourceURL(
            resource: resource, withExtension: "xml",
            subdirectory: subdirectory)
        return try parse(url: url)
    }

    /// The collision/dynamics MJCF from MuJoCo Menagerie's Unitree H1 model,
    /// vendored with its BSD-3-Clause attribution. Visual STL meshes are not
    /// needed by the physics importer.
    public static func bundledUnitreeH1() throws -> MJCFAsset {
        try bundled(resource: "h1", subdirectory: "Assets/unitree_h1")
    }

    public static func bundledArachne15(
        profile: Arachne15CollisionProfile = .training
    ) throws -> MJCFAsset {
        try bundled(resource: "arachne15_\(profile.rawValue)",
                    subdirectory: "Assets/arachne15")
    }

    /// Exact reduced 10-DoF plant used by Unitree RL Gym's public H1
    /// TorchScript/MuJoCo deployment, with analytic collision proxies.
    public static func bundledUnitreeRLGymH1() throws -> MJCFAsset {
        let url = try bundledResourceURL(
            resource: "unitree_rl_gym_h1", withExtension: "xml",
            subdirectory: "Assets/unitree_h1")
        return try parse(url: url)
    }

    /// AVBD's seven-axis Panda plant with an independently authored pusher.
    /// Robot frames, inertias, limits, passive joint parameters, and effort
    /// limits derive solely from the commit-pinned Apache-2.0 MuJoCo Menagerie
    /// model. AVBD-authored primitive proxies replace its collision meshes for
    /// the batched Metal contact path; `PROVENANCE.json` records exact hashes
    /// and every intentional transformation.
    public static func bundledPandaPusher() throws -> MJCFAsset {
        let url = try bundledResourceURL(
            resource: "panda_pusher", withExtension: "xml",
            subdirectory: "Assets/panda_pusher")
        return try parse(url: url)
    }

    /// Compatibility spelling retained for callers that identify the
    /// ManiSkill benchmark rather than the independently sourced robot asset.
    @available(*, deprecated, renamed: "bundledPandaPusher()")
    public static func bundledPandaStick() throws -> MJCFAsset {
        try bundledPandaPusher()
    }

    public static func parse(data: Data) throws -> MJCFAsset {
        try parse(
            data: data, baseURL: nil,
            visualNormalPolicy: .preserveImported)
    }

    private static func parse(
        data: Data, baseURL: URL?,
        visualNormalPolicy: SurfaceNormalPolicy
    ) throws -> MJCFAsset {
        let tree = XMLTreeParser()
        guard tree.parse(data), let root = tree.root else {
            throw MJCFImportError.malformedXML(tree.errorMessage ?? "empty document")
        }
        guard root.name == "mujoco" else {
            throw MJCFImportError.missing("<mujoco> root")
        }

        var defaults: [String: DefaultBundle] = [:]
        for node in root.children where node.name == "default" {
            collectDefaults(node, inherited: DefaultBundle(), into: &defaults)
        }

        var warnings: [String] = []
        let visualMaterials = try loadVisualMaterials(
            root: root, warnings: &warnings)
        let visualMeshes = loadVisualMeshes(
            root: root, baseURL: baseURL,
            normalPolicy: visualNormalPolicy, warnings: &warnings)

        guard let world = root.children.first(where: { $0.name == "worldbody" }) else {
            throw MJCFImportError.missing("<worldbody>")
        }
        var links: [Link] = []
        var bodyNameSet = Set<String>()
        var jointNameSet = Set<String>()
        for body in world.children where body.name == "body" {
            try parseBody(body, parent: nil, inheritedClass: nil,
                          defaults: defaults,
                          visualMaterials: visualMaterials,
                          visualMeshes: visualMeshes,
                          links: &links,
                          bodyNameSet: &bodyNameSet,
                          jointNameSet: &jointNameSet, warnings: &warnings)
        }
        guard !links.isEmpty else {
            throw MJCFImportError.missing("at least one articulated <body>")
        }

        let jointEffortLimits = Dictionary(uniqueKeysWithValues:
            links.compactMap { link -> (String, Float)? in
                guard let joint = link.joint,
                      let effortLimit = joint.actuatorForceLimit else {
                    return nil
                }
                return (joint.name, effortLimit)
            })
        var actuators: [Actuator] = []
        if let actuatorNode = root.children.first(where: { $0.name == "actuator" }) {
            for motor in actuatorNode.children where motor.name == "motor" {
                let joint = try required(motor, "joint")
                let name = motor.attributes["name"] ?? joint
                let forceLimit = try rangeMagnitude(
                    motor.attributes["forcerange"], element: "motor",
                    attribute: "forcerange")
                let controlLimit = try rangeMagnitude(
                    motor.attributes["ctrlrange"], element: "motor",
                    attribute: "ctrlrange")
                actuators.append(Actuator(name: name, joint: joint,
                    effortLimit: forceLimit
                        ?? jointEffortLimits[joint]
                        ?? controlLimit
                        ?? 1))
            }
        }

        var exclusions: [(String, String)] = []
        if let contact = root.children.first(where: { $0.name == "contact" }) {
            for exclude in contact.children where exclude.name == "exclude" {
                exclusions.append((try required(exclude, "body1"),
                                   try required(exclude, "body2")))
            }
        }

        for a in actuators where !jointNameSet.contains(a.joint) {
            throw MJCFImportError.missing("motor \(a.name) references joint \(a.joint)")
        }
        for (a, b) in exclusions
            where !bodyNameSet.contains(a) || !bodyNameSet.contains(b) {
            throw MJCFImportError.missing("contact exclusion references \(a), \(b)")
        }

        warnings = Array(Set(warnings)).sorted()
        return MJCFAsset(
            name: root.attributes["model"] ?? "mjcf",
            bodyNames: links.map(\.name),
            jointNames: links.compactMap { $0.joint?.name },
            actuatorNames: actuators.map(\.name),
            warnings: warnings,
            visualGeometryCount: links.reduce(0) {
                $0 + $1.visualGeometries.count
            },
            links: links, actuators: actuators, exclusions: exclusions)
    }

    /// Resolve source joint coordinates to solver body (COM/principal-frame)
    /// poses without constructing another scene. Batched task resets use this
    /// to apply authored joint-position randomization immediately rather than
    /// asking the motors to settle from a different pose after reset.
    public func bodyPoses(
        worldOffset: F3 = .zero,
        jointPositions: [String: Float] = [:],
        inertiaFrame: MJCFInertiaFrame = .principal
    ) throws -> [String: MJCFBodyPose] {
        var linkWorldP = [F3](repeating: .zero, count: links.count)
        var linkWorldQ = [Quat](repeating: identityQuaternion,
                                count: links.count)
        var result: [String: MJCFBodyPose] = [:]
        result.reserveCapacity(links.count)
        for (i, link) in links.enumerated() {
            var localPosition = link.localPosition
            var localRotation = link.localRotation
            if let joint = link.joint,
               let angle = jointPositions[joint.name], angle != 0 {
                guard angle >= joint.range.0 && angle <= joint.range.1 else {
                    throw MJCFImportError.invalidAttribute(
                        element: "joint", attribute: "position",
                        value: "\(joint.name)=\(angle)")
                }
                let delta = Quat(angle: angle, axis: joint.axis)
                localPosition += link.localRotation.act(
                    joint.position - delta.act(joint.position))
                localRotation = (link.localRotation * delta).normalized
            }
            if let parent = link.parent {
                linkWorldP[i] = linkWorldP[parent]
                    + linkWorldQ[parent].act(localPosition)
                linkWorldQ[i] = (linkWorldQ[parent] * localRotation).normalized
            } else {
                linkWorldP[i] = worldOffset + localPosition
                linkWorldQ[i] = localRotation
            }
            let principalRotation = inertiaFrame == .principal
                ? link.inertial.rotation : identityQuaternion
            result[link.name] = MJCFBodyPose(
                position: linkWorldP[i]
                    + linkWorldQ[i].act(link.inertial.position),
                rotation: (linkWorldQ[i] * principalRotation).normalized)
        }
        return result
    }

    /// Instantiate one copy. Rigid state is expressed in each source
    /// inertial/principal frame, not the link frame: COM offsets and inertial
    /// quaternions therefore remain exact while joints and colliders are
    /// transformed into that frame.
    @discardableResult
    public func instantiate(
        in scene: inout PhysicsScene,
        options: MJCFInstantiationOptions
    ) throws -> MJCFInstantiation {
        var linkWorldP = [F3](repeating: .zero, count: links.count)
        var linkWorldQ = [Quat](repeating: identityQuaternion, count: links.count)
        var bodyWorldP = [F3](repeating: .zero, count: links.count)
        var bodyWorldQ = [Quat](repeating: identityQuaternion, count: links.count)
        var bodyIndices = [Int](repeating: -1, count: links.count)
        var bodiesByName: [String: Int] = [:]
        var linkFramesInBody: [String: MJCFLinkFrame] = [:]

        for (i, link) in links.enumerated() {
            var posedLocalPosition = link.localPosition
            var posedLocalRotation = link.localRotation
            if let joint = link.joint,
               let home = options.jointHomePositions[joint.name], home != 0 {
                guard home >= joint.range.0 && home <= joint.range.1 else {
                    throw MJCFImportError.invalidAttribute(
                        element: "joint", attribute: "home",
                        value: "\(joint.name)=\(home)")
                }
                let delta = Quat(angle: home, axis: joint.axis)
                posedLocalPosition += link.localRotation.act(
                    joint.position - delta.act(joint.position))
                posedLocalRotation = (link.localRotation * delta).normalized
            }
            if let parent = link.parent {
                linkWorldP[i] = linkWorldP[parent]
                    + linkWorldQ[parent].act(posedLocalPosition)
                linkWorldQ[i] = (linkWorldQ[parent] * posedLocalRotation).normalized
            } else {
                linkWorldP[i] = options.worldOffset + posedLocalPosition
                linkWorldQ[i] = posedLocalRotation
            }
            bodyWorldP[i] = linkWorldP[i]
                + linkWorldQ[i].act(link.inertial.position)
            let inertialRotation = options.inertiaFrame == .principal
                ? link.inertial.rotation : identityQuaternion
            bodyWorldQ[i] = (linkWorldQ[i] * inertialRotation).normalized

            var reach: Float = 0.15
            for geom in link.geometries {
                reach = max(reach, length(geom.position - link.inertial.position)
                            + geom.boundingRadius)
            }
            let body = scene.addBody(
                size: F3(repeating: 2 * reach), density: 0,
                friction: (link.geometries.first?.friction ?? 1)
                    * options.dynamicsScale.friction,
                position: bodyWorldP[i], rotation: bodyWorldQ[i],
                mass: link.inertial.mass * options.dynamicsScale.mass,
                diagonalInertia: link.inertial.diagonalInertia
                    * options.dynamicsScale.mass
                    * options.dynamicsScale.inertia,
                gravityScale: options.gravityScale,
                collisionEnabled: false)
            bodyIndices[i] = body
            bodiesByName[link.name] = body

            let inertialInverse = inertialRotation.conjugate
            linkFramesInBody[link.name] = MJCFLinkFrame(
                position: inertialInverse.act(-link.inertial.position),
                rotation: inertialInverse)
            for geom in link.geometries {
                _ = scene.addCollider(
                    body: body, size: geom.size,
                    friction: geom.friction
                        * options.dynamicsScale.friction,
                    localPosition: inertialInverse.act(
                        geom.position - link.inertial.position),
                    localRotation: (inertialInverse * geom.rotation).normalized,
                    shape: geom.shape,
                    collisionGroup: options.collisionGroup,
                    collisionEnabled: geom.collisionEnabled,
                    // Imported CAD owns appearance when present. Contact
                    // proxies stay inspectable in assets without being drawn
                    // through the detailed surface.
                    isRendered: !options.includeVisuals
                        || link.visualGeometries.isEmpty,
                    renderColor: geom.renderColor)
            }
            for visual in options.includeVisuals
                ? link.visualGeometries : [] {
                let localPosition = inertialInverse.act(
                    visual.position - link.inertial.position)
                let localRotation = (inertialInverse
                    * visual.rotation).normalized
                switch visual.kind {
                case .mesh(let mesh):
                    scene.addRigidMesh(SceneRigidMesh(
                        body: body, mesh: mesh,
                        localPosition: localPosition,
                        localRotation: localRotation,
                        color: visual.color))
                case .primitive(let shape, let size):
                    _ = scene.addCollider(
                        body: body, size: size, friction: 0,
                        localPosition: localPosition,
                        localRotation: localRotation, shape: shape,
                        collisionGroup: options.collisionGroup,
                        collisionEnabled: false, isRendered: true,
                        renderColor: visual.color)
                }
            }
        }

        if options.fixedBase {
            scene.addJoint(SceneJoint(
                bodyA: -1, bodyB: bodyIndices[0],
                rA: bodyWorldP[0], rB: .zero,
                stiffnessLin: .infinity, stiffnessAng: .infinity))
        }

        let actuatorByJoint = Dictionary(uniqueKeysWithValues:
            actuators.map { ($0.joint, $0) })
        var jointsByName: [String: Int] = [:]
        for (i, link) in links.enumerated() {
            guard let parent = link.parent, let joint = link.joint else { continue }
            let anchor = linkWorldP[i] + linkWorldQ[i].act(joint.position)
            let parentQInverse = bodyWorldQ[parent].conjugate
            let childQInverse = bodyWorldQ[i].conjugate
            let rA = parentQInverse.act(anchor - bodyWorldP[parent])
            let rB = childQInverse.act(anchor - bodyWorldP[i])
            let axisWorld = linkWorldQ[i].act(joint.axis)
            let axisB = normalize(childQInverse.act(axisWorld))
            let actuator = actuatorByJoint[joint.name]
            let actuatorName = actuator?.name
            let gain = actuatorName.flatMap { options.motorGains[$0] }
                ?? options.motorGains[joint.name]
                ?? options.defaultMotorGain
            let effortLimit = actuatorName.flatMap {
                options.motorEffortLimits[$0]
            } ?? options.motorEffortLimits[joint.name]
                ?? actuator?.effortLimit
                ?? 0
            let armature = actuatorName.flatMap {
                options.jointArmatures[$0]
            } ?? options.jointArmatures[joint.name]
                ?? joint.armature
            if effortLimit > 0, gain.stiffness <= 0 {
                throw MJCFImportError.missing(
                    "positive position-PD stiffness for actuator "
                    + "\(actuatorName ?? joint.name) on joint \(joint.name)")
            }
            let home = options.jointHomePositions[joint.name] ?? 0
            let sceneJoint = SceneJoint(
                bodyA: bodyIndices[parent], bodyB: bodyIndices[i],
                rA: rA, rB: rB, stiffnessLin: .infinity,
                // A hinge still needs a hard two-axis angular constraint;
                // `hingeAxis` changes that constraint from a weld to axis
                // alignment and leaves only twist free for motor/limits.
                stiffnessAng: .infinity, hingeAxis: axisB,
                motorTarget: 0,
                motorTorque: effortLimit
                    * options.dynamicsScale.motorTorque,
                motorStiffness: gain.stiffness
                    * options.dynamicsScale.motorStiffness,
                // The fixed-PD actuator has one damping channel; treat an
                // explicitly supplied gain as the total and never undercut
                // passive source damping.
                motorDamping: max(gain.damping, joint.damping)
                    * options.dynamicsScale.motorDamping,
                armature: armature * options.dynamicsScale.armature,
                limitLo: joint.range.0 - home, limitHi: joint.range.1 - home)
            let index = scene.joints.count
            scene.addJoint(sceneJoint)
            jointsByName[joint.name] = index
        }

        for (a, b) in exclusions {
            guard let ia = bodiesByName[a], let ib = bodiesByName[b] else {
                throw MJCFImportError.missing("instantiated exclusion \(a), \(b)")
            }
            scene.addCollisionExclusion(bodyA: ia, bodyB: ib)
        }
        if !options.selfCollisions {
            for a in 0..<bodyIndices.count {
                for b in (a + 1)..<bodyIndices.count {
                    scene.addCollisionExclusion(
                        bodyA: bodyIndices[a], bodyB: bodyIndices[b])
                }
            }
        } else if !options.collideConnectedBodies {
            for (child, link) in links.enumerated() {
                guard let parent = link.parent else { continue }
                scene.addCollisionExclusion(
                    bodyA: bodyIndices[parent], bodyB: bodyIndices[child])
            }
        }
        let actuatorJoints = try actuators.map { actuator -> Int in
            guard let joint = jointsByName[actuator.joint] else {
                throw MJCFImportError.missing("instantiated actuator joint \(actuator.joint)")
            }
            return joint
        }
        return MJCFInstantiation(
            rootBody: bodyIndices[0], bodiesByName: bodiesByName,
            jointsByName: jointsByName, actuatorJoints: actuatorJoints,
            actuatorNames: actuators.map(\.name), warnings: warnings,
            linkFramesInBody: linkFramesInBody)
    }

    /// Compatibility overload for existing call sites. New code should pass
    /// an `MJCFInstantiationOptions` value so the complete plant setup is
    /// visible and reusable as one configuration.
    @available(*, deprecated, message: "Use instantiate(in:options:)")
    @discardableResult
    public func instantiate(
        in scene: inout PhysicsScene,
        worldOffset: F3 = .zero,
        defaultMotorGain: MJCFMotorGain = .init(stiffness: 0, damping: 0),
        motorGains: [String: MJCFMotorGain] = [:],
        jointHomePositions: [String: Float] = [:],
        motorEffortLimits: [String: Float] = [:],
        jointArmatures: [String: Float] = [:],
        fixedBase: Bool = false,
        gravityScale: Float = 1,
        collisionGroup: UInt32 = 0,
        selfCollisions: Bool = true,
        collideConnectedBodies: Bool = true,
        inertiaFrame: MJCFInertiaFrame = .principal,
        dynamicsScale: MJCFDynamicsScale = .identity,
        includeVisuals: Bool = true
    ) throws -> MJCFInstantiation {
        try instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                worldOffset: worldOffset,
                defaultMotorGain: defaultMotorGain,
                motorGains: motorGains,
                jointHomePositions: jointHomePositions,
                motorEffortLimits: motorEffortLimits,
                jointArmatures: jointArmatures,
                fixedBase: fixedBase,
                gravityScale: gravityScale,
                collisionGroup: collisionGroup,
                selfCollisions: selfCollisions,
                collideConnectedBodies: collideConnectedBodies,
                inertiaFrame: inertiaFrame,
                dynamicsScale: dynamicsScale,
                includeVisuals: includeVisuals))
    }
}

// MARK: - Parsed model

private let identityQuaternion = Quat(real: 1, imag: .zero)

private struct Link {
    var name: String
    var parent: Int?
    var localPosition: F3
    var localRotation: Quat
    var inertial: Inertial
    var joint: Joint?
    var geometries: [CollisionGeometry]
    var visualGeometries: [VisualGeometry]
}

private struct Inertial {
    var position: F3
    var rotation: Quat
    var mass: Float
    var diagonalInertia: F3
}

private struct Joint {
    var name: String
    var position: F3
    var axis: F3
    var range: (Float, Float)
    var damping: Float
    var armature: Float
    var actuatorForceLimit: Float?
}

private struct CollisionGeometry {
    var shape: BodyShape
    var size: F3
    var position: F3
    var rotation: Quat
    var friction: Float
    var boundingRadius: Float
    var collisionEnabled: Bool
    /// Authored `rgba` or named material, when the geom has one. Nil keeps
    /// the renderer's per-body fallback for uncolored contact geometry.
    var renderColor: F3?
}

private enum VisualGeometryKind {
    case mesh(SurfaceMesh)
    case primitive(BodyShape, F3)
}

private struct VisualGeometry {
    var kind: VisualGeometryKind
    var position: F3
    var rotation: Quat
    var color: F3
}

private struct Actuator {
    var name: String
    var joint: String
    var effortLimit: Float
}

private struct DefaultBundle {
    var geom: [String: String] = [:]
    var joint: [String: String] = [:]
}

private func collectDefaults(_ node: XMLNode, inherited: DefaultBundle,
                             into defaults: inout [String: DefaultBundle]) {
    var bundle = inherited
    for child in node.children {
        switch child.name {
        case "geom": bundle.geom.merge(child.attributes) { _, new in new }
        case "joint": bundle.joint.merge(child.attributes) { _, new in new }
        default: break
        }
    }
    if let className = node.attributes["class"] { defaults[className] = bundle }
    for child in node.children where child.name == "default" {
        collectDefaults(child, inherited: bundle, into: &defaults)
    }
}

private func parseBody(
    _ node: XMLNode, parent: Int?, inheritedClass: String?,
    defaults: [String: DefaultBundle], visualMaterials: [String: F3],
    visualMeshes: [String: SurfaceMesh],
    links: inout [Link],
    bodyNameSet: inout Set<String>, jointNameSet: inout Set<String>,
    warnings: inout [String]
) throws {
    let name = try required(node, "name")
    guard bodyNameSet.insert(name).inserted else {
        throw MJCFImportError.unsupported("duplicate body name \(name)")
    }
    let localPosition = try vector3(node.attributes["pos"] ?? "0 0 0",
                                    element: "body", attribute: "pos")
    let localRotation = try quaternion(node.attributes["quat"] ?? "1 0 0 0",
                                       element: "body", attribute: "quat")
    let activeClass = node.attributes["childclass"] ?? inheritedClass

    guard let inertialNode = node.children.first(where: { $0.name == "inertial" }) else {
        throw MJCFImportError.missing("body \(name) requires explicit <inertial>")
    }
    let inertial = Inertial(
        position: try vector3(inertialNode.attributes["pos"] ?? "0 0 0",
                              element: "inertial", attribute: "pos"),
        rotation: try quaternion(inertialNode.attributes["quat"] ?? "1 0 0 0",
                                 element: "inertial", attribute: "quat"),
        mass: try scalar(try required(inertialNode, "mass"),
                         element: "inertial", attribute: "mass"),
        diagonalInertia: try vector3(try required(inertialNode, "diaginertia"),
                                     element: "inertial", attribute: "diaginertia"))

    let jointNodes = node.children.filter {
        $0.name == "joint" || $0.name == "freejoint"
    }
    if jointNodes.count > 1 {
        throw MJCFImportError.unsupported("body \(name) has multiple joints")
    }
    var joint: Joint?
    if let jointNode = jointNodes.first {
        let jointClass = jointNode.attributes["class"] ?? activeClass
        var attrs = jointClass.flatMap { defaults[$0]?.joint } ?? [:]
        attrs.merge(jointNode.attributes) { _, new in new }
        let type = jointNode.name == "freejoint"
            ? "free" : attrs["type"] ?? "hinge"
        if type == "free" {
            guard parent == nil else {
                throw MJCFImportError.unsupported(
                    "free joint on \(name) must be on a top-level body")
            }
            // AVBD root bodies already carry a free six-DoF pose. A source
            // root free joint therefore requires no solver constraint and is
            // intentionally absent from the policy hinge ordering.
            joint = nil
        } else if type == "hinge" {
            let jointName = attrs["name"] ?? "\(name)_joint"
            guard jointNameSet.insert(jointName).inserted else {
                throw MJCFImportError.unsupported("duplicate joint name \(jointName)")
            }
            let rangeValues = try floatList(
                attrs["range"] ?? "-3.1415927 3.1415927",
                count: 2, element: "joint", attribute: "range")
            joint = Joint(
                name: jointName,
                position: try vector3(attrs["pos"] ?? "0 0 0",
                                      element: "joint", attribute: "pos"),
                axis: normalize(try vector3(attrs["axis"] ?? "0 0 1",
                                            element: "joint", attribute: "axis")),
                range: (rangeValues[0], rangeValues[1]),
                damping: try scalar(attrs["damping"] ?? "0",
                                    element: "joint", attribute: "damping"),
                armature: try scalar(attrs["armature"] ?? "0",
                                     element: "joint", attribute: "armature"),
                actuatorForceLimit: try rangeMagnitude(
                    attrs["actuatorfrcrange"], element: "joint",
                    attribute: "actuatorfrcrange"))
        } else {
            throw MJCFImportError.unsupported("joint type \(type) on \(name)")
        }
    }

    var geometries: [CollisionGeometry] = []
    var visualGeometries: [VisualGeometry] = []
    for geomNode in node.children where geomNode.name == "geom" {
        let className = geomNode.attributes["class"] ?? activeClass
        var attrs = className.flatMap { defaults[$0]?.geom } ?? [:]
        attrs.merge(geomNode.attributes) { _, new in new }
        let isVisualOnly = attrs["contype"] == "0"
            && attrs["conaffinity"] == "0"
        if attrs["type"] == "mesh" {
            if let meshName = attrs["mesh"],
               let mesh = visualMeshes[meshName] {
                visualGeometries.append(try parseVisualGeometry(
                    attrs, kind: .mesh(mesh), materials: visualMaterials))
            } else if let meshName = attrs["mesh"] {
                warnings.append("visual mesh \(meshName) is unavailable")
            }
            continue
        }
        if isVisualOnly {
            let primitive = try parseCollisionGeometry(
                attrs, bodyName: name, materials: visualMaterials,
                warnings: &warnings)
            visualGeometries.append(try parseVisualGeometry(
                attrs, kind: .primitive(primitive.shape, primitive.size),
                materials: visualMaterials,
                position: primitive.position, rotation: primitive.rotation))
            continue
        }
        geometries.append(try parseCollisionGeometry(
            attrs, bodyName: name, materials: visualMaterials,
            warnings: &warnings))
    }

    let index = links.count
    links.append(Link(name: name, parent: parent,
                      localPosition: localPosition, localRotation: localRotation,
                      inertial: inertial, joint: joint, geometries: geometries,
                      visualGeometries: visualGeometries))
    for child in node.children where child.name == "body" {
        try parseBody(child, parent: index, inheritedClass: activeClass,
                      defaults: defaults,
                      visualMaterials: visualMaterials,
                      visualMeshes: visualMeshes,
                      links: &links,
                      bodyNameSet: &bodyNameSet, jointNameSet: &jointNameSet,
                      warnings: &warnings)
    }
}

private func parseCollisionGeometry(
    _ attrs: [String: String], bodyName: String, materials: [String: F3],
    warnings: inout [String]
) throws -> CollisionGeometry {
    let type = attrs["type"] ?? "sphere"
    let renderColor = try authoredColor(attrs, materials: materials)
    let friction = try floatList(attrs["friction"] ?? "1 0.005 0.0001",
                                 minimumCount: 1, element: "geom",
                                 attribute: "friction")[0]
    let collisionEnabled = attrs["contype"] != "0"
        && attrs["conaffinity"] != "0"
    var position = try vector3(attrs["pos"] ?? "0 0 0",
                               element: "geom", attribute: "pos")
    var rotation = try quaternion(attrs["quat"] ?? "1 0 0 0",
                                  element: "geom", attribute: "quat")
    let rawSize = try floatList(attrs["size"] ?? "",
                                minimumCount: 1, element: "geom", attribute: "size")
    let shape: BodyShape
    let size: F3
    let radius: Float

    switch type {
    case "box":
        guard rawSize.count >= 3 else {
            throw MJCFImportError.invalidAttribute(
                element: "geom", attribute: "size", value: attrs["size"] ?? "")
        }
        shape = .box
        size = 2 * F3(rawSize[0], rawSize[1], rawSize[2])
        radius = length(size * 0.5)
    case "sphere":
        shape = .sphere
        size = F3(repeating: 2 * rawSize[0])
        radius = rawSize[0]
    case "capsule", "cylinder":
        let r = rawSize[0]
        var length: Float
        if let fromTo = attrs["fromto"] {
            let v = try floatList(fromTo, count: 6,
                                  element: "geom", attribute: "fromto")
            let a = F3(v[0], v[1], v[2])
            let b = F3(v[3], v[4], v[5])
            let d = b - a
            length = simd_length(d)
            guard length > 1e-8 else {
                throw MJCFImportError.invalidAttribute(
                    element: "geom", attribute: "fromto", value: fromTo)
            }
            position = (a + b) * 0.5
            rotation = rotationFromZ(to: d / length)
        } else {
            guard rawSize.count >= 2 else {
                throw MJCFImportError.invalidAttribute(
                    element: "geom", attribute: "size", value: attrs["size"] ?? "")
            }
            length = 2 * rawSize[1]
        }
        // AVBD currently has a robust segment+sphere capsule contact path.
        // Cylinder source geoms are conservatively represented by that shape
        // and reported; H1's ground-contact feet are native capsules.
        if type == "cylinder" {
            warnings.append("cylinder collision geoms are conservatively represented as capsules")
        }
        shape = .capsule
        size = F3(length, r, 0)
        radius = length * 0.5 + r
    default:
        throw MJCFImportError.unsupported(
            "collision geom type \(type) on body \(bodyName)")
    }
    return CollisionGeometry(shape: shape, size: size, position: position,
                             rotation: rotation, friction: friction,
                             boundingRadius: radius,
                             collisionEnabled: collisionEnabled,
                             renderColor: renderColor)
}

private func parseVisualGeometry(
    _ attrs: [String: String], kind: VisualGeometryKind,
    materials: [String: F3],
    position: F3? = nil, rotation: Quat? = nil
) throws -> VisualGeometry {
    // Retain AVBD's established unmaterialed-CAD fallback.
    let color = try authoredColor(attrs, materials: materials)
        ?? F3(0.24, 0.28, 0.34)
    return VisualGeometry(
        kind: kind,
        position: try position ?? vector3(
            attrs["pos"] ?? "0 0 0", element: "geom", attribute: "pos"),
        rotation: try rotation ?? quaternion(
            attrs["quat"] ?? "1 0 0 0", element: "geom", attribute: "quat"),
        color: color)
}

/// The color a geom authors, or nil when it has neither `rgba` nor a
/// `material`. Shared by rendered collision primitives and visual-only geoms
/// so a named material colors an ordinary MJCF geom that both collides and
/// renders, not only one with zeroed contact masks.
private func authoredColor(
    _ attrs: [String: String], materials: [String: F3]
) throws -> F3? {
    if let rawRGBA = attrs["rgba"] {
        // MuJoCo gives a local/default-class rgba precedence over a referenced
        // material. Preserve that rule instead of blindly applying the asset.
        let rgba = try colorComponents(
            rawRGBA, element: "geom", attribute: "rgba")
        return F3(rgba[0], rgba[1], rgba[2])
    }
    if let name = attrs["material"] {
        guard let material = materials[name] else {
            throw MJCFImportError.missing(
                "geom references material \(name)")
        }
        return material
    }
    return nil
}

/// Resolve named MJCF materials once at the asset boundary. SurfaceMesh is a
/// position/normal STL representation and therefore has no UV channel; models
/// that reference a texture keep their material tint and report the omitted
/// texture explicitly rather than silently turning black.
private func loadVisualMaterials(
    root: XMLNode, warnings: inout [String]
) throws -> [String: F3] {
    guard let assets = root.children.first(where: { $0.name == "asset" }) else {
        return [:]
    }
    var result: [String: F3] = [:]
    for node in assets.children where node.name == "material" {
        let name = try required(node, "name")
        guard result[name] == nil else {
            throw MJCFImportError.unsupported(
                "duplicate visual material \(name)")
        }
        let rgba = try colorComponents(
            node.attributes["rgba"] ?? "1 1 1 1",
            element: "material", attribute: "rgba")
        result[name] = F3(rgba[0], rgba[1], rgba[2])
        if node.attributes["texture"] != nil {
            warnings.append(
                "visual material \(name) texture is unavailable; using rgba")
        }
    }
    return result
}

private func colorComponents(
    _ value: String, element: String, attribute: String
) throws -> [Float] {
    let rgba = try floatList(
        value, count: 4, element: element, attribute: attribute)
    guard rgba.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
        throw MJCFImportError.invalidAttribute(
            element: element, attribute: attribute, value: value)
    }
    return rgba
}

/// Load available visual assets without making rendering geometry a physics
/// requirement. A package may intentionally vendor only dynamics/collision
/// MJCF (the H1 asset does); missing visuals become explicit warnings while
/// the articulation remains usable headlessly.
private func loadVisualMeshes(
    root: XMLNode, baseURL: URL?, normalPolicy: SurfaceNormalPolicy,
    warnings: inout [String]
) -> [String: SurfaceMesh] {
    guard let assets = root.children.first(where: { $0.name == "asset" }) else {
        return [:]
    }
    let meshDirectory = root.children.first(where: { $0.name == "compiler" })?
        .attributes["meshdir"] ?? ""
    var result: [String: SurfaceMesh] = [:]
    for node in assets.children where node.name == "mesh" {
        guard let name = node.attributes["name"],
              let file = node.attributes["file"] else {
            warnings.append("visual <mesh> is missing name or file")
            continue
        }
        guard let baseURL else {
            warnings.append("visual mesh \(name) skipped without an asset base URL")
            continue
        }
        let url = baseURL.appendingPathComponent(meshDirectory)
            .appendingPathComponent(file).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            warnings.append("visual mesh \(name) not bundled at \(file)")
            continue
        }
        do {
            let scale = try vector3(node.attributes["scale"] ?? "1 1 1",
                                    element: "mesh", attribute: "scale")
            let source = try SurfaceMesh.load(
                path: url.path, upAxis: .z, normalPolicy: normalPolicy)
            let positions = source.vertices.map { $0 * scale }
            let inverseScale = F3(
                1 / max(abs(scale.x), Float.leastNormalMagnitude),
                1 / max(abs(scale.y), Float.leastNormalMagnitude),
                1 / max(abs(scale.z), Float.leastNormalMagnitude))
            let normals = source.normals.map { normalize($0 * inverseScale) }
            result[name] = SurfaceMesh(
                vertices: positions, normals: normals,
                triangles: source.triangles)
        } catch {
            warnings.append("visual mesh \(name) failed to load: \(error)")
        }
    }
    return result
}

private func rotationFromZ(to direction: F3) -> Quat {
    let z = F3(0, 0, 1)
    let c = dot(z, direction)
    if c > 0.999999 { return identityQuaternion }
    if c < -0.999999 { return Quat(angle: .pi, axis: F3(1, 0, 0)) }
    return Quat(real: 1 + c, imag: cross(z, direction)).normalized
}

// MARK: - XML and numeric helpers

private final class XMLNode {
    let name: String
    let attributes: [String: String]
    var children: [XMLNode] = []

    init(name: String, attributes: [String: String]) {
        self.name = name
        self.attributes = attributes
    }
}

private final class XMLTreeParser: NSObject, XMLParserDelegate {
    var root: XMLNode?
    var stack: [XMLNode] = []
    var errorMessage: String?

    func parse(_ data: Data) -> Bool {
        let parser = XMLParser(data: data)
        parser.delegate = self
        let ok = parser.parse()
        if !ok { errorMessage = parser.parserError?.localizedDescription }
        return ok
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?,
                attributes attributeDict: [String: String] = [:]) {
        let node = XMLNode(name: elementName, attributes: attributeDict)
        if let parent = stack.last { parent.children.append(node) } else { root = node }
        stack.append(node)
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?) {
        _ = stack.popLast()
    }
}

private func required(_ node: XMLNode, _ attribute: String) throws -> String {
    guard let value = node.attributes[attribute], !value.isEmpty else {
        throw MJCFImportError.missing("<\(node.name)> attribute \(attribute)")
    }
    return value
}

private func scalar(_ value: String, element: String,
                    attribute: String) throws -> Float {
    guard let result = Float(value), result.isFinite else {
        throw MJCFImportError.invalidAttribute(
            element: element, attribute: attribute, value: value)
    }
    return result
}

private func rangeMagnitude(
    _ value: String?, element: String, attribute: String
) throws -> Float? {
    guard let value else { return nil }
    let range = try floatList(value, count: 2, element: element,
                              attribute: attribute)
    return max(abs(range[0]), abs(range[1]))
}

private func floatList(_ value: String, count: Int, element: String,
                       attribute: String) throws -> [Float] {
    let result = try floatList(value, minimumCount: count,
                               element: element, attribute: attribute)
    guard result.count == count else {
        throw MJCFImportError.invalidAttribute(
            element: element, attribute: attribute, value: value)
    }
    return result
}

private func floatList(_ value: String, minimumCount: Int, element: String,
                       attribute: String) throws -> [Float] {
    let tokens = value.split(whereSeparator: { $0.isWhitespace })
    let result = tokens.compactMap { Float($0) }
    guard result.count == tokens.count, result.count >= minimumCount,
          result.allSatisfy(\.isFinite) else {
        throw MJCFImportError.invalidAttribute(
            element: element, attribute: attribute, value: value)
    }
    return result
}

private func vector3(_ value: String, element: String,
                     attribute: String) throws -> F3 {
    let v = try floatList(value, count: 3, element: element, attribute: attribute)
    return F3(v[0], v[1], v[2])
}

private func quaternion(_ value: String, element: String,
                        attribute: String) throws -> Quat {
    let v = try floatList(value, count: 4, element: element, attribute: attribute)
    let q = Quat(real: v[0], imag: F3(v[1], v[2], v[3]))
    guard q.length > 1e-8 else {
        throw MJCFImportError.invalidAttribute(
            element: element, attribute: attribute, value: value)
    }
    return q.normalized
}
