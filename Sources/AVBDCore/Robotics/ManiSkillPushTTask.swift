import simd

/// Faithful state-based port of ManiSkill PushT-v1. This task intentionally
/// lives beside the historical two-axis ArmPushT task so old checkpoints keep
/// their original embodiment and signatures.
public struct ManiSkillPushTTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
    public var autoReset: Bool
    public var jointDeltaActionScale: Float
    public var robotInitialJointNoise: Float
    public var normalizedDenseReward: Bool

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        maxEpisodeSteps: Int = 100, controlDecimation: Int = 5,
        autoReset: Bool = true, jointDeltaActionScale: Float = 0.1,
        robotInitialJointNoise: Float = 0.02,
        normalizedDenseReward: Bool = true
    ) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
        self.autoReset = autoReset
        self.jointDeltaActionScale = jointDeltaActionScale
        self.robotInitialJointNoise = robotInitialJointNoise
        self.normalizedDenseReward = normalizedDenseReward
    }
}

public struct PandaStickPushTState {
    public var tcpPosition: F3
    public var tcpRotation: Quat
    public var tcpLinearVelocity: F3
    public var blockPosition: F3
    public var blockRotation: Quat
    public var blockLinearVelocity: F3
    public var blockAngularVelocity: F3
    public var jointPositions: [Float]
    public var jointVelocities: [Float]
}

public struct PandaStickTCPPose {
    public var position: F3
    public var rotation: Quat

    public init(position: F3, rotation: Quat) {
        self.position = position
        self.rotation = rotation
    }
}

/// Batched seven-axis Panda with AVBD's pusher, executing the ManiSkill
/// PushT-v1 scene/reward contract. The robot asset has separate Menagerie-only
/// provenance; this benchmark task intentionally retains ManiSkill-derived
/// reset, workpiece, controller, observation, reward, and success constants.
public final class PandaStickPushTEnv {
    public struct EnvRefs {
        public var center: F3
        public var robotBodies: [Int]
        public var motors: [Int]
        public var link7: Int
        public var link7Frame: MJCFLinkFrame
        public var block: Int
        public var goalPosition: SIMD2<Float>
        public var goalYaw: Float
    }

    public static let actionCount = 7
    public static let defaultJointPositions: [Float] = [
        0.662, 0.212, 0.086, -2.685, -0.115, 2.898, 1.673,
    ]
    public static let jointRanges: [(Float, Float)] = [
        (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
        (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ]
    public static let basePosition = F3(-0.615, 0, 0)
    /// Distal cap of AVBD's independently authored pusher in link7 space.
    public static let tcpInLink7 = F3(0, 0, 0.255)
    public static let goalPosition = SIMD2<Float>(-0.156, -0.1)
    public static let goalYaw: Float = 5 * .pi / 3
    public static let successCoverage: Float = 0.90
    public static let blockMass: Float = 0.8
    public static let blockFriction: Float = 3
    public static let environmentSpacing: Float = 1
    public static let barSize = F3(0.20, 0.05, 0.04)
    public static let stemSize = F3(0.05, 0.15, 0.04)
    private static let blockArea: Float = 0.20 * 0.05 + 0.05 * 0.15
    /// Uniform-density inertia of ManiSkill's two-box T about its authored
    /// center of mass. SAPIEN builds both collision boxes on one rigid actor;
    /// using the same compound representation avoids weld compliance and
    /// duplicated integration/contact state in AVBD.
    public static let blockDiagonalInertia: F3 = {
        let barMass = blockMass * 4 / 7
        let stemMass = blockMass * 3 / 7
        func boxInertia(mass: Float, size: F3, yOffset: Float) -> F3 {
            F3(
                mass * (size.y * size.y + size.z * size.z) / 12
                    + mass * yOffset * yOffset,
                mass * (size.x * size.x + size.z * size.z) / 12,
                mass * (size.x * size.x + size.y * size.y) / 12
                    + mass * yOffset * yOffset)
        }
        return boxInertia(mass: barMass, size: barSize, yOffset: -0.0375)
            + boxInertia(mass: stemMass, size: stemSize, yOffset: 0.0625)
    }()

    private static let pseudoRenderResolution = 64
    private static let pseudoRenderScale: Float = 32 / 0.15
    /// Exact source pixels produced by ManiSkill's 64x64 `tee_render` setup.
    /// Coordinates are the physical UV values that its code transforms from
    /// the actor frame into the fixed goal frame.
    private static let pseudoRenderSourcePixels: [SIMD2<Float>] = {
        var points: [SIMD2<Float>] = []
        points.reserveCapacity(825)
        for row in 0..<pseudoRenderResolution {
            for column in 0..<pseudoRenderResolution
            where pseudoRenderGoalContains(row: row, column: column) {
                points.append(SIMD2(
                    (Float(column) - 31.5) / pseudoRenderScale,
                    (32.5 - Float(row)) / pseudoRenderScale))
            }
        }
        return points
    }()
    private static let pseudoRenderGoalRows: [UInt64] = {
        (0..<pseudoRenderResolution).map { row in
            var bits: UInt64 = 0
            for column in 0..<pseudoRenderResolution
            where pseudoRenderGoalContains(row: row, column: column) {
                bits |= UInt64(1) << UInt64(column)
            }
            return bits
        }
    }()
    private static let pseudoRenderGoalArea = 825

    public let numEnvironments: Int
    public let scene: PhysicsScene
    public let solver: GPUSolver
    public let model: MJCFAsset
    public private(set) var refs: [EnvRefs]
    private let spawnPoses: [(F3, Quat)]

    public init(numEnvironments: Int, seed: UInt64 = 1) throws {
        precondition(numEnvironments > 0)
        self.numEnvironments = numEnvironments
        model = try MJCFAsset.bundledPandaPusher()
        var built = PhysicsScene(name: "maniskill-pusht-v1")
        // ManiSkill SimConfig defaults: 100 Hz simulation, 20 Hz control,
        // 15 position iterations, and 0.3/0.3 default surface friction.
        built.settings.dt = 1 / 100
        built.settings.iterations = 15
        // The legacy solver's 1 cm normal slop is appropriate for its
        // meter-scale demos but would let this 4 cm workpiece sink through a
        // quarter of its thickness. PhysX resolves toward zero rest offset;
        // 1 mm keeps speculative contact robust without changing task scale.
        built.settings.collisionMargin = 0.001
        built.settings.betaLin = 20_000
        built.settings.betaAng = 500
        built.settings.lambdaMax = 1_200
        // PhysX materials default to average combination. With ManiSkill's
        // 3.0 T material and 0.3 default table/stick material this produces
        // 1.65 effective friction; AVBD's legacy geometric mean produced only
        // 0.949 and let the stick slip during the rotational push phase.
        built.settings.frictionCombineMode = .average
        built.settings.cameraDistance = 1.15
        built.settings.cameraTargetZ = 0.12
        let ground = Demos.addGround(&built, friction: 0.3)
        let side = Int(ceil(Double(numEnvironments).squareRoot()))
        var rng = SplitMix64(seed: seed)
        var allRefs: [EnvRefs] = []
        allRefs.reserveCapacity(numEnvironments)
        for environment in 0..<numEnvironments {
            let center = F3(
                Float(environment % side) * Self.environmentSpacing,
                Float(environment / side) * Self.environmentSpacing, 0)
            allRefs.append(try Self.buildOne(
                model: model, scene: &built, center: center,
                ground: ground, rng: &rng,
                showGoal: numEnvironments <= 4))
        }
        refs = allRefs
        scene = built
        spawnPoses = built.bodies.map { ($0.position, $0.rotation) }
        solver = try GPUSolver(scene: built)
    }

    private static var homePositionsByName: [String: Float] {
        Dictionary(uniqueKeysWithValues: (0..<actionCount).map {
            ("joint\($0 + 1)", defaultJointPositions[$0])
        })
    }

    private static func buildOne(
        model: MJCFAsset, scene: inout PhysicsScene, center: F3,
        ground: Int, rng: inout SplitMix64, showGoal: Bool
    ) throws -> EnvRefs {
        let collisionGroup = UInt32(scene.bodies.count + 1)
        let instance = try model.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                worldOffset: center + basePosition,
                defaultMotorGain: MJCFMotorGain(
                    stiffness: 1_000, damping: 100),
                jointHomePositions: homePositionsByName,
                fixedBase: true,
                gravityScale: 0,
                collisionGroup: collisionGroup,
                selfCollisions: false))
        let robotBodies = try model.bodyNames.map { name in
            guard let body = instance.bodiesByName[name] else {
                throw MJCFImportError.missing("Panda body \(name)")
            }
            return body
        }
        // The articulation root is fixed to the table/world. Generating a
        // second hard contact between that fixed root and the static table is
        // both redundant and unlike PhysX's fixed-base articulation filtering;
        // at finite solver iterations it can stretch the seven-link chain.
        scene.addCollisionExclusion(bodyA: ground, bodyB: instance.rootBody)
        guard let link7 = instance.bodiesByName["link7"],
              let link7Frame = instance.linkFramesInBody["link7"] else {
            throw MJCFImportError.missing("Panda link7 semantic frame")
        }

        let goal = goalPosition
        let x = goal.x - 0.1 + 0.2 * rng.nextFloat()
        let y = goal.y - 0.1 + 0.3 * rng.nextFloat()
        let yaw = 2 * .pi * rng.nextFloat()
        let q = Quat(angle: yaw, axis: F3(0, 0, 1))
        let blockCenter = center + F3(x, y, 0.021)
        let block = scene.addBody(
            size: barSize, density: 0, friction: blockFriction,
            dynamicFriction: blockFriction,
            position: blockCenter, rotation: q,
            mass: blockMass, diagonalInertia: blockDiagonalInertia,
            collisionEnabled: false)
        scene.addCollider(
            body: block, size: barSize,
            localPosition: F3(0, -0.0375, 0),
            collisionGroup: collisionGroup)
        scene.addCollider(
            body: block, size: stemSize,
            localPosition: F3(0, 0.0625, 0),
            collisionGroup: collisionGroup)

        if showGoal {
            let goalQ = Quat(angle: goalYaw, axis: F3(0, 0, 1))
            let goalCenter = center + F3(goal.x, goal.y, 0.0001)
            _ = scene.addBody(
                size: F3(0.20, 0.05, 0.0002), density: 0, friction: 0,
                position: goalCenter + goalQ.act(F3(0, -0.0375, 0)),
                rotation: goalQ, collisionEnabled: false)
            _ = scene.addBody(
                size: F3(0.05, 0.15, 0.0002), density: 0, friction: 0,
                position: goalCenter + goalQ.act(F3(0, 0.0625, 0)),
                rotation: goalQ, collisionEnabled: false)
        }
        return EnvRefs(
            center: center, robotBodies: robotBodies,
            motors: instance.actuatorJoints, link7: link7,
            link7Frame: link7Frame, block: block,
            goalPosition: goal, goalYaw: goalYaw)
    }

    public func step(actions: ContiguousArray<Float>, decimation: Int,
                     deltaScale: Float) {
        precondition(actions.count == numEnvironments * Self.actionCount)
        precondition(decimation > 0 && deltaScale > 0)
        let motorIDs = refs.flatMap(\.motors)
        let measured = solver.motorAngles(motorIDs)
        var updates: [GPUSolver.MotorTargetUpdate] = []
        updates.reserveCapacity(actions.count)
        for environment in 0..<numEnvironments {
            for joint in 0..<Self.actionCount {
                let index = environment * Self.actionCount + joint
                let range = Self.jointRanges[joint]
                let sourceAngle = Self.defaultJointPositions[joint]
                    + measured[index]
                let requested = simd_clamp(actions[index], -1, 1)
                let targetSource = simd_clamp(
                    sourceAngle + requested * deltaScale,
                    range.0, range.1)
                updates.append(.init(
                    joint: motorIDs[index],
                    angle: targetSource - Self.defaultJointPositions[joint]))
            }
        }
        solver.setMotorTargets(updates)
        for _ in 0..<decimation { solver.step() }
    }

    /// Resolve the task-local tool-tip pose from source Panda joint angles.
    /// This is intentionally the same semantic TCP transform used by
    /// `states()`, making offline IK and the simulated observation agree even
    /// though rigid bodies are integrated in their inertial frames.
    public func kinematicTCPPose(
        jointPositions: [Float], environment: Int = 0
    ) throws -> PandaStickTCPPose {
        precondition(jointPositions.count == Self.actionCount)
        precondition(environment >= 0 && environment < numEnvironments)
        let sourceJoints = Dictionary(uniqueKeysWithValues:
            (0..<Self.actionCount).map {
                ("joint\($0 + 1)", jointPositions[$0])
            })
        let ref = refs[environment]
        let poses = try model.bodyPoses(
            worldOffset: ref.center + Self.basePosition,
            jointPositions: sourceJoints)
        guard let body = poses["link7"] else {
            throw MJCFImportError.missing("kinematic link7 pose")
        }
        let linkRotation = (body.rotation * ref.link7Frame.rotation).normalized
        let linkPosition = body.position
            + body.rotation.act(ref.link7Frame.position)
        return PandaStickTCPPose(
            position: linkPosition + linkRotation.act(Self.tcpInLink7)
                - ref.center,
            rotation: linkRotation)
    }

    /// Damped-least-squares Cartesian IK used by deterministic diagnostics and
    /// demonstration generation. Learned policies never call this routine.
    /// The optional orientation target is valuable for the pusher because a
    /// position-only solution can tilt the 11 cm tool into the table or T.
    public func inverseKinematics(
        targetPosition: F3, targetRotation: Quat? = nil,
        initialJointPositions: [Float] = defaultJointPositions,
        environment: Int = 0, maximumIterations: Int = 120,
        positionTolerance: Float = 5e-4,
        rotationTolerance: Float = 0.015
    ) throws -> [Float] {
        precondition(initialJointPositions.count == Self.actionCount)
        precondition(maximumIterations > 0 && positionTolerance > 0
                     && rotationTolerance > 0)
        var q = initialJointPositions
        let orientationWeight: Float = targetRotation == nil ? 0 : 0.20
        let finiteDifference: Float = 1e-3
        let damping: Float = 0.025

        for _ in 0..<maximumIterations {
            let current = try kinematicTCPPose(
                jointPositions: q, environment: environment)
            let positionError = targetPosition - current.position
            let rotationError = targetRotation.map {
                Self.rotationVector($0 * current.rotation.conjugate)
            } ?? .zero
            if length(positionError) <= positionTolerance,
               targetRotation == nil || length(rotationError) <= rotationTolerance {
                return q
            }

            var error = [Float](repeating: 0, count: 6)
            error[0] = positionError.x
            error[1] = positionError.y
            error[2] = positionError.z
            error[3] = orientationWeight * rotationError.x
            error[4] = orientationWeight * rotationError.y
            error[5] = orientationWeight * rotationError.z
            var jacobian = [Float](repeating: 0, count: 6 * Self.actionCount)
            for joint in 0..<Self.actionCount {
                let range = Self.jointRanges[joint]
                var perturbed = q
                var delta = min(finiteDifference, range.1 - q[joint])
                if delta < finiteDifference * 0.5 {
                    delta = -min(finiteDifference, q[joint] - range.0)
                }
                guard abs(delta) > 1e-7 else { continue }
                perturbed[joint] += delta
                let pose = try kinematicTCPPose(
                    jointPositions: perturbed, environment: environment)
                let dp = (pose.position - current.position) / delta
                let dr = Self.rotationVector(
                    pose.rotation * current.rotation.conjugate) / delta
                jacobian[joint] = dp.x
                jacobian[Self.actionCount + joint] = dp.y
                jacobian[2 * Self.actionCount + joint] = dp.z
                jacobian[3 * Self.actionCount + joint] = orientationWeight * dr.x
                jacobian[4 * Self.actionCount + joint] = orientationWeight * dr.y
                jacobian[5 * Self.actionCount + joint] = orientationWeight * dr.z
            }

            // dq = J^T (J J^T + lambda^2 I)^-1 error. Solving the 6x6
            // task-space system is cheaper and better conditioned than a
            // generic 7x7 inverse for this redundant arm.
            var normal = [Float](repeating: 0, count: 36)
            for row in 0..<6 {
                for column in 0..<6 {
                    var sum: Float = 0
                    for joint in 0..<Self.actionCount {
                        sum += jacobian[row * Self.actionCount + joint]
                            * jacobian[column * Self.actionCount + joint]
                    }
                    normal[row * 6 + column] = sum
                }
                normal[row * 6 + row] += damping * damping
            }
            guard let taskStep = Self.solveLinear6(normal, error) else { break }
            var jointStep = [Float](repeating: 0, count: Self.actionCount)
            var largest: Float = 0
            for joint in 0..<Self.actionCount {
                for row in 0..<6 {
                    jointStep[joint] += jacobian[row * Self.actionCount + joint]
                        * taskStep[row]
                }
                largest = max(largest, abs(jointStep[joint]))
            }
            let scale = largest > 0.20 ? 0.20 / largest : 1
            for joint in 0..<Self.actionCount {
                q[joint] = simd_clamp(
                    q[joint] + scale * jointStep[joint],
                    Self.jointRanges[joint].0, Self.jointRanges[joint].1)
            }
        }
        return q
    }

    private static func rotationVector(_ raw: Quat) -> F3 {
        var real = raw.real
        var imaginary = raw.imag
        // q and -q encode the same orientation. The positive-real branch is
        // the shortest rotation and keeps finite-difference columns smooth.
        if real < 0 {
            real = -real
            imaginary = -imaginary
        }
        let sineHalf = length(imaginary)
        guard sineHalf > 1e-8 else { return 2 * imaginary }
        return imaginary / sineHalf * (2 * atan2(sineHalf, real))
    }

    private static func solveLinear6(
        _ coefficients: [Float], _ rightHandSide: [Float]
    ) -> [Float]? {
        precondition(coefficients.count == 36 && rightHandSide.count == 6)
        var a = coefficients
        var b = rightHandSide
        for pivot in 0..<6 {
            var best = pivot
            for row in (pivot + 1)..<6
            where abs(a[row * 6 + pivot]) > abs(a[best * 6 + pivot]) {
                best = row
            }
            guard abs(a[best * 6 + pivot]) > 1e-10 else { return nil }
            if best != pivot {
                for column in pivot..<6 {
                    a.swapAt(pivot * 6 + column, best * 6 + column)
                }
                b.swapAt(pivot, best)
            }
            let divisor = a[pivot * 6 + pivot]
            for column in pivot..<6 { a[pivot * 6 + column] /= divisor }
            b[pivot] /= divisor
            for row in 0..<6 where row != pivot {
                let factor = a[row * 6 + pivot]
                guard factor != 0 else { continue }
                for column in pivot..<6 {
                    a[row * 6 + column] -= factor * a[pivot * 6 + column]
                }
                b[row] -= factor * b[pivot]
            }
        }
        return b
    }

    public func reset(_ environmentIDs: [Int], seeds: [UInt64],
                      jointNoise: Float) throws {
        precondition(environmentIDs.count == seeds.count && jointNoise >= 0)
        var poses: [GPUSolver.BodyPoseUpdate] = []
        var motors: [GPUSolver.MotorTargetUpdate] = []
        for (offset, environment) in environmentIDs.enumerated() {
            var rng = SplitMix64(seed: seeds[offset])
            let ref = refs[environment]
            var sourceJoints: [String: Float] = [:]
            for joint in 0..<Self.actionCount {
                let noise = jointNoise * Self.standardNormal(&rng)
                let range = Self.jointRanges[joint]
                let source = simd_clamp(
                    Self.defaultJointPositions[joint] + noise,
                    range.0, range.1)
                sourceJoints["joint\(joint + 1)"] = source
                motors.append(.init(
                    joint: ref.motors[joint],
                    angle: source - Self.defaultJointPositions[joint]))
            }
            let robotPoses = try model.bodyPoses(
                worldOffset: ref.center + Self.basePosition,
                jointPositions: sourceJoints)
            for (bodyOffset, name) in model.bodyNames.enumerated() {
                guard let pose = robotPoses[name] else {
                    throw MJCFImportError.missing("reset pose for \(name)")
                }
                poses.append(.init(
                    body: ref.robotBodies[bodyOffset],
                    position: pose.position, rotation: pose.rotation))
            }

            let x = ref.goalPosition.x - 0.1 + 0.2 * rng.nextFloat()
            let y = ref.goalPosition.y - 0.1 + 0.3 * rng.nextFloat()
            let yaw = 2 * .pi * rng.nextFloat()
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let center = ref.center + F3(x, y, 0.021)
            poses.append(.init(body: ref.block, position: center, rotation: q))
        }
        solver.setBodyPoses(poses)
        solver.setMotorTargets(motors)
    }

    /// Teleport one Panda articulation to an authored joint configuration.
    /// This is a deterministic diagnostic/reset primitive: every rigid link,
    /// velocity, motor target, and incident warm start is updated together so
    /// controller/contact experiments do not inherit a previous trajectory.
    public func setRobotJointPositions(
        _ jointPositions: [Float], environment: Int = 0
    ) throws {
        precondition(jointPositions.count == Self.actionCount)
        precondition(environment >= 0 && environment < numEnvironments)
        var sourceJoints: [String: Float] = [:]
        var motors: [GPUSolver.MotorTargetUpdate] = []
        let ref = refs[environment]
        for joint in 0..<Self.actionCount {
            let source = simd_clamp(
                jointPositions[joint], Self.jointRanges[joint].0,
                Self.jointRanges[joint].1)
            sourceJoints["joint\(joint + 1)"] = source
            motors.append(.init(
                joint: ref.motors[joint],
                angle: source - Self.defaultJointPositions[joint]))
        }
        let robotPoses = try model.bodyPoses(
            worldOffset: ref.center + Self.basePosition,
            jointPositions: sourceJoints)
        let updates = try model.bodyNames.enumerated().map { offset, name in
            guard let pose = robotPoses[name] else {
                throw MJCFImportError.missing("diagnostic pose for \(name)")
            }
            return GPUSolver.BodyStateUpdate(
                body: ref.robotBodies[offset], position: pose.position,
                rotation: pose.rotation)
        }
        solver.setBodyStates(updates)
        solver.setMotorTargets(motors)
    }

    public func states() -> [PandaStickPushTState] {
        let bodyIDs = refs.flatMap { [$0.link7, $0.block] }
        let bodies = solver.bodyStates(bodyIDs)
        let motorStates = solver.motorStates(refs.flatMap(\.motors))
        return (0..<numEnvironments).map { environment in
            let ref = refs[environment]
            let link7 = bodies[environment * 2]
            let block = bodies[environment * 2 + 1]
            let linkRotation = (link7.rotation * ref.link7Frame.rotation).normalized
            let linkPosition = link7.position
                + link7.rotation.act(ref.link7Frame.position)
            let tcpOffset = linkRotation.act(Self.tcpInLink7)
            let tcpPosition = linkPosition + tcpOffset
            let tcpVelocity = link7.linearVelocity
                + cross(link7.angularVelocity, tcpPosition - link7.position)
            let base = environment * Self.actionCount
            return PandaStickPushTState(
                tcpPosition: tcpPosition - ref.center,
                tcpRotation: linkRotation,
                tcpLinearVelocity: tcpVelocity,
                blockPosition: block.position - ref.center,
                blockRotation: block.rotation,
                blockLinearVelocity: block.linearVelocity,
                blockAngularVelocity: block.angularVelocity,
                jointPositions: (0..<Self.actionCount).map {
                    Self.defaultJointPositions[$0] + motorStates[base + $0].angle
                },
                jointVelocities: (0..<Self.actionCount).map {
                    motorStates[base + $0].velocity
                })
        }
    }

    public func coverage(_ environment: Int,
                         state: PandaStickPushTState) -> Float {
        let forward = state.blockRotation.act(F3(1, 0, 0))
        return Self.coverage(
            blockPosition: SIMD2(state.blockPosition.x, state.blockPosition.y),
            blockYaw: atan2(forward.y, forward.x),
            goalPosition: refs[environment].goalPosition,
            goalYaw: refs[environment].goalYaw)
    }

    public static func coverage(
        blockPosition: SIMD2<Float>, blockYaw: Float,
        goalPosition: SIMD2<Float>, goalYaw: Float
    ) -> Float {
        let block = rectangles(position: blockPosition, yaw: blockYaw)
        let goal = rectangles(position: goalPosition, yaw: goalYaw)
        var area: Float = 0
        for subject in block {
            for clip in goal { area += polygonArea(clipPolygon(subject, by: clip)) }
        }
        return simd_clamp(area / blockArea, 0, 1)
    }

    /// ManiSkill PushT-v1's success metric, including its authored 64x64 UV
    /// grid, integer truncation, scatter deduplication, transpose, and Y flip.
    /// This deliberately does not substitute continuous polygon overlap: the
    /// benchmark terminates episodes using this discrete pseudo-render score.
    public static func officialPseudoRenderCoverage(
        blockPosition: SIMD2<Float>, blockYaw: Float,
        goalPosition: SIMD2<Float>, goalYaw: Float
    ) -> Float {
        let relativeYaw = blockYaw - goalYaw
        let c = cos(relativeYaw), s = sin(relativeYaw)
        let worldDelta = blockPosition - goalPosition
        let gc = cos(goalYaw), gs = sin(goalYaw)
        let relativePosition = SIMD2(
            gc * worldDelta.x + gs * worldDelta.y,
            -gs * worldDelta.x + gc * worldDelta.y)
        return withUnsafeTemporaryAllocation(
            of: UInt64.self, capacity: pseudoRenderResolution
        ) { rendered in
            rendered.initialize(repeating: 0)
            for point in pseudoRenderSourcePixels {
                let transformed = relativePosition + SIMD2(
                    c * point.x - s * point.y,
                    s * point.x + c * point.y)
                // Torch `.long()` truncates toward zero. In-range image
                // coordinates are positive, so Swift's Int conversion is
                // identical. Out-of-range coordinates are discarded; the
                // upstream implementation maps them to [0,0], which is not a
                // goal-T pixel and therefore cannot increase intersection.
                let x = Int(transformed.x * pseudoRenderScale + 32)
                let y = Int(transformed.y * pseudoRenderScale + 32)
                guard x >= 0, x < pseudoRenderResolution,
                      y >= 0, y < pseudoRenderResolution else { continue }
                let row = 63 - y
                rendered[row] |= UInt64(1) << UInt64(x)
            }
            var intersection = 0
            for row in 0..<pseudoRenderResolution {
                intersection += (rendered[row] & pseudoRenderGoalRows[row])
                    .nonzeroBitCount
            }
            return Float(intersection) / Float(pseudoRenderGoalArea)
        }
    }

    private static func pseudoRenderGoalContains(
        row: Int, column: Int
    ) -> Bool {
        // Exact slices after ManiSkill transposes `tee_render` and flips Y:
        // box1.T[10..<53, 18..<29], box2.T[26..<37, 29..<61].
        (row >= 35 && row < 46 && column >= 10 && column < 53)
            || (row >= 3 && row < 35 && column >= 26 && column < 37)
    }

    private static func rectangles(position: SIMD2<Float>, yaw: Float)
        -> [[SIMD2<Float>]] {
        [rectangle(position: position, localCenter: SIMD2(0, -0.0375),
                   halfExtents: SIMD2(0.1, 0.025), yaw: yaw),
         rectangle(position: position, localCenter: SIMD2(0, 0.0625),
                   halfExtents: SIMD2(0.025, 0.075), yaw: yaw)]
    }

    private static func rectangle(
        position: SIMD2<Float>, localCenter: SIMD2<Float>,
        halfExtents: SIMD2<Float>, yaw: Float
    ) -> [SIMD2<Float>] {
        let c = cos(yaw), s = sin(yaw)
        func transform(_ p: SIMD2<Float>) -> SIMD2<Float> {
            let q = p + localCenter
            return position + SIMD2(c * q.x - s * q.y, s * q.x + c * q.y)
        }
        let h = halfExtents
        return [transform(SIMD2(-h.x, -h.y)), transform(SIMD2(h.x, -h.y)),
                transform(SIMD2(h.x, h.y)), transform(SIMD2(-h.x, h.y))]
    }

    private static func cross2(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        a.x * b.y - a.y * b.x
    }

    private static func clipPolygon(
        _ subject: [SIMD2<Float>], by clip: [SIMD2<Float>]
    ) -> [SIMD2<Float>] {
        var output = subject
        for index in clip.indices {
            let a = clip[index]
            let edge = clip[(index + 1) % clip.count] - a
            let input = output
            output.removeAll(keepingCapacity: true)
            guard !input.isEmpty else { break }
            for pointIndex in input.indices {
                let p = input[pointIndex]
                let q = input[(pointIndex + 1) % input.count]
                let pSide = cross2(edge, p - a)
                let qSide = cross2(edge, q - a)
                let pInside = pSide >= -1e-6
                let qInside = qSide >= -1e-6
                if pInside != qInside {
                    let denominator = pSide - qSide
                    if abs(denominator) > 1e-8 {
                        output.append(p + (q - p) * (pSide / denominator))
                    }
                }
                if qInside { output.append(q) }
            }
        }
        return output
    }

    private static func polygonArea(_ polygon: [SIMD2<Float>]) -> Float {
        guard polygon.count >= 3 else { return 0 }
        var twiceArea: Float = 0
        for index in polygon.indices {
            twiceArea += cross2(
                polygon[index], polygon[(index + 1) % polygon.count])
        }
        return abs(twiceArea) * 0.5
    }

    private static func standardNormal(_ rng: inout SplitMix64) -> Float {
        let u1 = max(rng.nextFloat(), 1e-7)
        let u2 = rng.nextFloat()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}

public final class ManiSkillPushTTask: VectorizedRLTask,
    RLEvaluationCriteriaProviding
{
    public let spec: RLTaskSpec
    public let environment: PandaStickPushTEnv
    public let configuration: ManiSkillPushTTaskConfig
    public let evaluationCriteria = RLEvaluationCriteria(
        minimumSuccessRate: 0.80,
        minimumTaskMetrics: ["episode/normalized_score": 0.80])

    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var episodeSucceeded: [Bool]
    private var maximumScores: [Float]
    private var resetRNG: SplitMix64

    public init(configuration: ManiSkillPushTTaskConfig) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.controlDecimation > 0,
              configuration.jointDeltaActionScale > 0,
              configuration.robotInitialJointNoise >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "ManiSkill PushT dimensions and controller scales must be positive")
        }
        self.configuration = configuration
        environment = try PandaStickPushTEnv(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed)
        spec = RLTaskSpec(
            // Revision 9 replaces the noncommercial PandaStick asset with the
            // pinned Menagerie Panda, Menagerie's effort/sliding-friction
            // semantics,
            // and AVBD's independently authored 110 mm pusher. The explicit
            // revision boundary rejects revision-8 checkpoints rather than
            // silently replaying them against a different physical plant.
            // Revision 8 preserved the former 100 mm cylindrical stick's
            // total collision envelope when representing it as a capsule.
            // Revision 7 uses a millimeter-scale contact margin for the thin
            // T/stick instead of the engine's meter-scale 1 cm legacy slop.
            // Revision 6 fixes the mounted-base collision proxy/filter so the
            // physical articulation reaches its observed Cartesian pose.
            // Revision 5 uses PhysX's default average friction combination.
            // Revision 4 matches SAPIEN's single compound T rigid actor and
            // uses ManiSkill's exact 64x64 pseudo-render success metric.
            // Revision 3 represented the two T boxes as separately integrated
            // bodies connected by an AVBD weld and used continuous overlap.
            id: "maniskill-pusht-v1",
            revision: RLPhysicsContract.fixedGainActuatorV2(9),
            numEnvironments: configuration.numEnvironments,
            // ManiSkill state observation: qpos, qvel, TCP pose, goal
            // position, and object pose = 7 + 7 + 7 + 3 + 7 = 31.
            observation: RLTensorSpec(name: "state", shape: [31]),
            action: RLTensorSpec(
                name: "pd_joint_delta_position", shape: [7],
                lowerBound: [Float](repeating: -1, count: 7),
                upperBound: [Float](repeating: 1, count: 7)),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: environment.scene.settings.dt,
            controlDecimation: configuration.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: [
                "jointDeltaActionScale": configuration.jointDeltaActionScale,
                "robotInitialJointNoise": configuration.robotInitialJointNoise,
                "normalizedDenseReward": configuration.normalizedDenseReward ? 1 : 0,
            ])
        let count = configuration.numEnvironments
        episodeLengths = [Int](repeating: 0, count: count)
        episodeReturns = [Float](repeating: 0, count: count)
        episodeSucceeded = [Bool](repeating: false, count: count)
        maximumScores = [Float](repeating: 0, count: count)
        resetRNG = SplitMix64(seed: configuration.seed &+ 0xDF11A0C553A1D7B9)
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let environmentIDs = try checkedEnvironmentIDs(ids)
        let seeds = environmentIDs.map {
            seed &+ UInt64($0) &* 0x9E3779B97F4A7C15
        }
        try environment.reset(
            environmentIDs, seeds: seeds,
            jointNoise: configuration.robotInitialJointNoise)
        initializeEpisodes(environmentIDs)
        fillObservations(environment.states(), into: &observations.policy)
    }

    public func step(actions: RLActionBatch,
                     into result: inout RLStepBatch) throws {
        try actions.validate(for: spec)
        try result.validate(for: spec)
        result.clearSignals()
        environment.step(
            actions: actions.values,
            decimation: configuration.controlDecimation,
            deltaScale: configuration.jointDeltaActionScale)
        var states = environment.states()
        fillObservations(states, into: &result.observations.policy)
        let count = spec.numEnvironments
        var normalizedScore = ContiguousArray(repeating: Float(0), count: count)
        var continuousCoverageMetric = normalizedScore
        var goalDistanceMetric = normalizedScore
        var yawErrorMetric = normalizedScore
        var tcpDistanceMetric = normalizedScore
        var episodeReturnMetric = normalizedScore
        var episodeLengthMetric = normalizedScore
        var episodeSuccessMetric = normalizedScore
        var episodeScoreMetric = normalizedScore
        var episodeOfficialCoverageTerminalMetric = normalizedScore
        var episodeContinuousCoverageTerminalMetric = normalizedScore
        var episodeGoalDistanceTerminalMetric = normalizedScore
        var episodeYawErrorTerminalMetric = normalizedScore
        var episodeTCPDistanceTerminalMetric = normalizedScore
        var resetIDs: [Int] = []
        var resetSeeds: [UInt64] = []
        for environmentIndex in 0..<count {
            let state = states[environmentIndex]
            let goal = environment.refs[environmentIndex].goalPosition
            let blockXY = SIMD2(state.blockPosition.x, state.blockPosition.y)
            let goalDistance = length(blockXY - goal)
            let forward = state.blockRotation.act(F3(1, 0, 0))
            var yawError = atan2(forward.y, forward.x)
                - environment.refs[environmentIndex].goalYaw
            yawError -= 2 * .pi * (yawError / (2 * .pi)).rounded()
            let forwardYaw = atan2(forward.y, forward.x)
            let continuousCoverage = environment.coverage(
                environmentIndex, state: state)
            let officialCoverage = PandaStickPushTEnv
                .officialPseudoRenderCoverage(
                    blockPosition: blockXY, blockYaw: forwardYaw,
                    goalPosition: goal,
                    goalYaw: environment.refs[environmentIndex].goalYaw)
            let score = min(
                officialCoverage / PandaStickPushTEnv.successCoverage, 1)
            let success = officialCoverage >= configurationSuccessCoverage
            let tcpDistance = length(state.blockPosition - state.tcpPosition)
            let rawReward = Self.denseReward(
                goalDistance: goalDistance, yawError: yawError,
                tcpDistance: tcpDistance,
                success: success)
            let reward = configuration.normalizedDenseReward
                ? rawReward / 3 : rawReward
            result.rewards[environmentIndex] = reward
            normalizedScore[environmentIndex] = score
            continuousCoverageMetric[environmentIndex] = continuousCoverage
            goalDistanceMetric[environmentIndex] = goalDistance
            yawErrorMetric[environmentIndex] = abs(yawError)
            tcpDistanceMetric[environmentIndex] = tcpDistance
            episodeLengths[environmentIndex] += 1
            episodeReturns[environmentIndex] += reward
            episodeSucceeded[environmentIndex] =
                episodeSucceeded[environmentIndex] || success
            maximumScores[environmentIndex] = max(
                maximumScores[environmentIndex], score)
            let timeout = episodeLengths[environmentIndex]
                >= configuration.maxEpisodeSteps
            result.terminated[environmentIndex] = success
            result.truncated[environmentIndex] = !success && timeout
            result.successes[environmentIndex] = success
            if success || timeout {
                result.hasFinalObservation[environmentIndex] = true
                let row = environmentIndex * spec.observation.elementCount
                for column in 0..<spec.observation.elementCount {
                    result.finalObservations[row + column] =
                        result.observations.policy[row + column]
                }
                episodeReturnMetric[environmentIndex] =
                    episodeReturns[environmentIndex]
                episodeLengthMetric[environmentIndex] =
                    Float(episodeLengths[environmentIndex])
                episodeSuccessMetric[environmentIndex] =
                    episodeSucceeded[environmentIndex] ? 1 : 0
                episodeScoreMetric[environmentIndex] =
                    maximumScores[environmentIndex]
                episodeOfficialCoverageTerminalMetric[environmentIndex] =
                    officialCoverage
                episodeContinuousCoverageTerminalMetric[environmentIndex] =
                    continuousCoverage
                episodeGoalDistanceTerminalMetric[environmentIndex] =
                    goalDistance
                episodeYawErrorTerminalMetric[environmentIndex] = abs(yawError)
                episodeTCPDistanceTerminalMetric[environmentIndex] = tcpDistance
                if configuration.autoReset {
                    resetIDs.append(environmentIndex)
                    resetSeeds.append(resetRNG.next())
                }
            }
        }
        result.metrics["task/normalized_coverage"] = normalizedScore
        result.metrics["task/continuous_coverage"] = continuousCoverageMetric
        result.metrics["task/goal_distance_m"] = goalDistanceMetric
        result.metrics["task/yaw_error_rad"] = yawErrorMetric
        result.metrics["task/tcp_distance_m"] = tcpDistanceMetric
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/success_once"] = episodeSuccessMetric
        result.metrics["episode/normalized_score"] = episodeScoreMetric
        result.metrics["episode/official_coverage_terminal"] =
            episodeOfficialCoverageTerminalMetric
        result.metrics["episode/continuous_coverage_terminal"] =
            episodeContinuousCoverageTerminalMetric
        result.metrics["episode/goal_distance_m_terminal"] =
            episodeGoalDistanceTerminalMetric
        result.metrics["episode/yaw_error_rad_terminal"] =
            episodeYawErrorTerminalMetric
        result.metrics["episode/tcp_distance_m_terminal"] =
            episodeTCPDistanceTerminalMetric
        if !resetIDs.isEmpty {
            try environment.reset(
                resetIDs, seeds: resetSeeds,
                jointNoise: configuration.robotInitialJointNoise)
            initializeEpisodes(resetIDs)
            states = environment.states()
            fillObservations(states, into: &result.observations.policy)
        }
    }

    private var configurationSuccessCoverage: Float {
        PandaStickPushTEnv.successCoverage
    }

    private func initializeEpisodes(_ ids: [Int]) {
        for environment in ids {
            episodeLengths[environment] = 0
            episodeReturns[environment] = 0
            episodeSucceeded[environment] = false
            maximumScores[environment] = 0
        }
    }

    private func fillObservations(
        _ states: [PandaStickPushTState],
        into output: inout ContiguousArray<Float>
    ) {
        let width = spec.observation.elementCount
        for environment in 0..<spec.numEnvironments {
            let state = states[environment]
            let base = environment * width
            for joint in 0..<7 {
                output[base + joint] = state.jointPositions[joint]
                output[base + 7 + joint] = state.jointVelocities[joint]
            }
            output[base + 14] = state.tcpPosition.x
            output[base + 15] = state.tcpPosition.y
            output[base + 16] = state.tcpPosition.z
            output[base + 17] = state.tcpRotation.real
            output[base + 18] = state.tcpRotation.imag.x
            output[base + 19] = state.tcpRotation.imag.y
            output[base + 20] = state.tcpRotation.imag.z
            let goal = environment < self.environment.refs.count
                ? self.environment.refs[environment].goalPosition : .zero
            output[base + 21] = goal.x
            output[base + 22] = goal.y
            output[base + 23] = 0.001
            output[base + 24] = state.blockPosition.x
            output[base + 25] = state.blockPosition.y
            output[base + 26] = state.blockPosition.z
            output[base + 27] = state.blockRotation.real
            output[base + 28] = state.blockRotation.imag.x
            output[base + 29] = state.blockRotation.imag.y
            output[base + 30] = state.blockRotation.imag.z
        }
    }

    public static func denseReward(
        goalDistance: Float, yawError: Float,
        tcpDistance: Float, success: Bool
    ) -> Float {
        if success { return 3 }
        let rotation = 0.5 * pow(0.5 * (cos(yawError) + 1), 2)
        let translation = 0.5 * pow(1 - tanh(5 * goalDistance), 2)
        let reaching = sqrt(max(1 - tanh(5 * tcpDistance), 0)) / 20
        return rotation + translation + reaching
    }
}
