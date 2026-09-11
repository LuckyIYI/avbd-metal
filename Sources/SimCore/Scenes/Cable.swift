import simd

/// Resolution-independent elastic rod parameters in SI units. Dividing each
/// rigidity by the adjacent segments' mean length gives a joint stiffness.
public struct CableMaterial: Sendable, Equatable {
    public let stretchRigidity: Float // EA, N
    public let shearRigidity: Float   // GA, N
    public let bendRigidity: Float    // EI, N m²
    public let twistRigidity: Float   // GJ, N m²
    /// Kelvin–Voigt relaxation time (seconds): damping = stiffness * time.
    public let dampingTime: Float

    public init(stretchRigidity: Float, shearRigidity: Float,
                bendRigidity: Float, twistRigidity: Float,
                dampingTime: Float = 0) {
        precondition(stretchRigidity > 0 && stretchRigidity.isFinite
            && shearRigidity > 0 && shearRigidity.isFinite)
        precondition(bendRigidity >= 0 && bendRigidity.isFinite
            && twistRigidity >= 0 && twistRigidity.isFinite
            && dampingTime >= 0 && dampingTime.isFinite)
        self.stretchRigidity = stretchRigidity
        self.shearRigidity = shearRigidity
        self.bendRigidity = bendRigidity
        self.twistRigidity = twistRigidity
        self.dampingTime = dampingTime
    }

    /// Homogeneous circular cross section (unit shear correction factor).
    public static func circular(radius: Float, youngModulus: Float,
                                poissonRatio: Float = 0.3,
                                dampingTime: Float = 0) -> Self {
        precondition(radius > 0 && radius.isFinite
            && youngModulus > 0 && youngModulus.isFinite
            && poissonRatio > -1 && poissonRatio < 0.5)
        let area = Float.pi * radius * radius
        let secondMoment = area * radius * radius / 4
        let shearModulus = youngModulus / (2 * (1 + poissonRatio))
        return Self(stretchRigidity: youngModulus * area,
                    shearRigidity: shearModulus * area,
                    bendRigidity: youngModulus * secondMoment,
                    twistRigidity: shearModulus * 2 * secondMoment,
                    dampingTime: dampingTime)
    }
}

/// Per-joint physical coefficients, captured when a cable is authored.
/// Linear coordinates are parent-local [shear, shear, stretch]; angular
/// coordinates are the rest-child-local rotation vector [bend, bend, twist].
public struct CableJointMaterial: Sendable, Equatable {
    public let linearStiffness: F3
    public let angularStiffness: F3
    public let dampingTime: Float

    public init(material: CableMaterial, restLength: Float) {
        precondition(restLength > 0 && restLength.isFinite)
        linearStiffness = F3(material.shearRigidity, material.shearRigidity,
                             material.stretchRigidity) / restLength
        angularStiffness = F3(material.bendRigidity, material.bendRigidity,
                              material.twistRigidity) / restLength
        precondition(linearStiffness.max().isFinite
            && angularStiffness.max().isFinite)
        dampingTime = material.dampingTime
    }
}

/// Stable authored topology. Bodies also remain ordinary solver body IDs,
/// so picking, state updates, attachments and rendering use the usual APIs.
public struct SceneCable: Sendable, Equatable {
    public let bodyIDs: [Int]
    public let jointIDs: [Int]
    public let restLengths: [Float]
    public let radius: Float

    public var startAnchor: F3 { F3(0, 0, -restLengths[0] / 2) }
    public var endAnchor: F3 { F3(0, 0, restLengths[restLengths.count - 1] / 2) }

    internal func remapped(body: (Int) -> Int, joint: (Int) -> Int) -> Self {
        Self(bodyIDs: bodyIDs.map(body), jointIDs: jointIDs.map(joint),
             restLengths: restLengths, radius: radius)
    }
}

public enum CableAuthoringError: Error, Equatable {
    case insufficientPoints
    case invalidGeometry
    case invalidFixedSegment(Int)
    case invalidMaterialFrame(Int)
}

public extension PhysicsScene {
    /// Build a stress-free open cable from a polyline. Each edge is a cylinder
    /// inertial body with a capsule collider; overlapping end caps add no mass.
    /// Adjacent links do not collide. Other links retain ordinary self-contact.
    /// Optional frames must align local +Z with each edge; omitted frames use
    /// minimal parallel transport to avoid introducing rest twist.
    @discardableResult
    mutating func addCable(points: [F3], radius: Float, density: Float,
                           material: CableMaterial, friction: Float = 0.5,
                           fixedSegments: Set<Int> = [],
                           rotations: [Quat]? = nil,
                           collisionGroup: UInt32 = 0,
                           collisionEnabled: Bool = true) throws -> SceneCable {
        guard points.count >= 2 else { throw CableAuthoringError.insufficientPoints }
        guard radius > 0 && radius.isFinite && density > 0 && density.isFinite
            && friction >= 0 && friction.isFinite
            && points.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
        else { throw CableAuthoringError.invalidGeometry }
        let count = points.count - 1
        for index in fixedSegments where !(0..<count).contains(index) {
            throw CableAuthoringError.invalidFixedSegment(index)
        }
        if let rotations, rotations.count != count {
            throw CableAuthoringError.invalidMaterialFrame(rotations.count)
        }
        // Validate everything before mutating the scene.
        var lengths: [Float] = [], frames: [Quat] = []
        lengths.reserveCapacity(count); frames.reserveCapacity(count)
        let area = Float.pi * radius * radius
        for i in 0..<count {
            let edge = points[i + 1] - points[i], l = length(edge)
            let mass = density * area * l
            let inertia = mass * F3((3 * radius * radius + l * l) / 12,
                                    (3 * radius * radius + l * l) / 12,
                                    radius * radius / 2)
            guard l > 1e-6 && l.isFinite && mass > 0 && mass.isFinite
                && inertia.min() > 0 && inertia.max().isFinite
            else { throw CableAuthoringError.invalidGeometry }
            let tangent = edge / l
            let frame: Quat
            if let rotations {
                let q = rotations[i]
                guard q.vector.x.isFinite && q.vector.y.isFinite
                    && q.vector.z.isFinite && q.vector.w.isFinite
                    && abs(length(q.vector) - 1) < 1e-4
                    && length(q.act(F3(0, 0, 1)) - tangent) < 1e-4
                else { throw CableAuthoringError.invalidMaterialFrame(i) }
                frame = q.normalized
            } else if let previous = frames.last {
                frame = (Quat(from: previous.act(F3(0, 0, 1)), to: tangent)
                         * previous).normalized
            } else {
                frame = Quat(from: F3(0, 0, 1), to: tangent)
            }
            lengths.append(l); frames.append(frame)
        }
        var jointMaterials: [CableJointMaterial] = []
        for i in 1..<count {
            let dualLength = (lengths[i - 1] + lengths[i]) * 0.5
            guard (max(material.stretchRigidity, material.shearRigidity) / dualLength).isFinite
                && (max(material.bendRigidity, material.twistRigidity) / dualLength).isFinite
            else { throw CableAuthoringError.invalidGeometry }
            jointMaterials.append(CableJointMaterial(material: material, restLength: dualLength))
        }
        var ids: [Int] = [], joints: [Int] = []
        ids.reserveCapacity(count); joints.reserveCapacity(count - 1)
        for i in 0..<count {
            let l = lengths[i]
            let mass = fixedSegments.contains(i) ? 0 : density * area * l
            let inertia = mass * F3((3 * radius * radius + l * l) / 12,
                                    (3 * radius * radius + l * l) / 12,
                                    radius * radius / 2)
            let body = addBody(size: F3(l, radius, 0), density: density,
                               friction: friction,
                               position: points[i] * 0.5 + points[i + 1] * 0.5,
                               rotation: frames[i], shape: .capsule,
                               mass: mass, diagonalInertia: inertia,
                               collisionEnabled: false)
            // Material anchors rotate with the segment, unlike the legacy
            // rotation-invariant standalone round-body friction convention.
            _ = addCollider(body: body, size: F3(l, radius, 0), shape: .capsule,
                            collisionGroup: collisionGroup,
                            collisionEnabled: collisionEnabled)
            ids.append(body)
            if i > 0 {
                var joint = SceneJoint(bodyA: ids[i - 1], bodyB: body,
                    rA: F3(0, 0, lengths[i - 1] / 2), rB: F3(0, 0, -l / 2),
                    stiffnessLin: 0, stiffnessAng: 0)
                joint.cable = jointMaterials[i - 1]
                joints.append(self.joints.count)
                addJoint(joint)
            }
        }
        let cable = SceneCable(bodyIDs: ids, jointIDs: joints,
                               restLengths: lengths, radius: radius)
        cables.append(cable)
        return cable
    }
}
