import simd

/// A deterministic, non-neural locomotion baseline for Arachne-15.
///
/// The controller never changes the root pose. A paired ripple oscillator
/// sweeps six stance feet behind the body while two feet follow smooth recovery
/// arcs; inverse kinematics turns those body-space foot targets into the same
/// bounded joint-position actions used by learned policies. Consequently all
/// motion still has to be produced by motor torque, contact, friction, and the
/// AVBD solver.
public final class Arachne15ClassicalController {
    public struct Configuration: Sendable, Equatable {
        /// Number of 50 Hz controller updates used to move one foot.
        public var swingSteps: Int
        /// Maximum vertical clearance of a swinging foot centre.
        public var swingHeight: Float
        /// Half-stride horizon used to convert desired body twist to travel.
        public var placementHorizon: Float
        /// Prevents unreachable placements from saturating the hip servos.
        public var maximumPlanarPlacement: Float
        /// Avoids asymptotically stalling outside a point-goal radius. This is
        /// a gait-speed floor, not a minimum command: zero still stands.
        public var minimumTranslationSpeed: Float
        /// Commands below this norm hold all feet instead of advancing phase.
        public var standingCommandThreshold: Float

        public init(
            swingSteps: Int = 4,
            swingHeight: Float = 0.016,
            placementHorizon: Float = 0.35,
            maximumPlanarPlacement: Float = 0.045,
            minimumTranslationSpeed: Float = 0.075,
            standingCommandThreshold: Float = 0.008
        ) {
            precondition(swingSteps >= 2 && swingHeight > 0
                && placementHorizon > 0 && maximumPlanarPlacement > 0
                && minimumTranslationSpeed >= 0
                && standingCommandThreshold >= 0)
            self.swingSteps = swingSteps
            self.swingHeight = swingHeight
            self.placementHorizon = placementHorizon
            self.maximumPlanarPlacement = maximumPlanarPlacement
            self.minimumTranslationSpeed = minimumTranslationSpeed
            self.standingCommandThreshold = standingCommandThreshold
        }
    }

    public struct Diagnostics: Sendable, Equatable {
        public var activeSwingLeg: [Int?]
        public var swingPhase: [Float]
        public var constrainedTargetCount: Int
    }

    /// Fixed geometry from the generated, measured Arachne-15 MJCF.
    public static let coxaLength: Float = 0.050
    public static let tibiaLength: Float = 0.105
    public static let fixedTibiaPitch: Float = 65 * .pi / 180
    public static let hipPositions: [F3] = [
        F3(-0.060, -0.092, 0.022), F3(-0.024, -0.092, 0.022),
        F3( 0.024, -0.092, 0.022), F3( 0.060, -0.092, 0.022),
        F3(-0.060,  0.092, 0.022), F3(-0.024,  0.092, 0.022),
        F3( 0.024,  0.092, 0.022), F3( 0.060,  0.092, 0.022),
    ]
    public static let fixedHipYaws: [Float] = [
        -140 * .pi / 180, -110 * .pi / 180,
         -70 * .pi / 180,  -40 * .pi / 180,
         140 * .pi / 180,  110 * .pi / 180,
          70 * .pi / 180,   40 * .pi / 180,
    ]

    /// Paired ripple gait. Each pair is separated both laterally and
    /// longitudinally, leaving a wide six-foot support polygon. Four phases
    /// complete a cycle without the yaw bias of a single-foot wave.
    public static let defaultWaveGroups = [
        [0, 6], [3, 5], [1, 7], [2, 4],
    ]

    private struct EnvironmentState {
        var neutralFeetInBody: [F3]
        var gaitStep = 0
        var moving = false
    }

    public let configuration: Configuration
    public private(set) var diagnostics: Diagnostics
    private var environments: [EnvironmentState] = []

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
        diagnostics = Diagnostics(
            activeSwingLeg: [], swingPhase: [], constrainedTargetCount: 0)
    }

    public func reset(states: [Arachne15State]) {
        environments = states.map { state in
            let inverse = state.root.rotation.conjugate
            let neutral = state.feet.map {
                inverse.act($0.position - state.root.position)
            }
            return EnvironmentState(
                neutralFeetInBody: neutral)
        }
        diagnostics = Diagnostics(
            activeSwingLeg: [Int?](repeating: nil, count: states.count),
            swingPhase: [Float](repeating: 0, count: states.count),
            constrainedTargetCount: 0)
    }

    /// Produce one batched action for the task's current measured state and
    /// body-frame twist commands.
    public func actions(
        states: [Arachne15State], commands: [F3], spec: RLTaskSpec
    ) -> RLActionBatch {
        precondition(states.count == commands.count
            && states.count == spec.numEnvironments
            && spec.action.elementCount == Arachne15PolicyContract.actionDimension)
        if environments.count != states.count { reset(states: states) }
        diagnostics.constrainedTargetCount = 0
        var batch = RLActionBatch(spec: spec)
        for e in states.indices {
            updateEnvironment(e, state: states[e], command: commands[e],
                              actions: &batch)
        }
        return batch
    }

    private func updateEnvironment(
        _ environment: Int, state: Arachne15State, command: F3,
        actions: inout RLActionBatch
    ) {
        var memory = environments[environment]
        let motionNorm = sqrt(command.x * command.x + command.y * command.y)
            + 0.08 * abs(command.z)
        if motionNorm < configuration.standingCommandThreshold {
            if memory.moving {
                memory.moving = false
                memory.gaitStep = 0
            }
            diagnostics.activeSwingLeg[environment] = nil
            diagnostics.swingPhase[environment] = 0
        } else {
            if !memory.moving {
                memory.moving = true
            }
        }

        let cycleSteps = configuration.swingSteps
            * Self.defaultWaveGroups.count
        var swingLeg: Int?
        var reportedSwingPhase: Float = 0
        for leg in 0..<8 {
            let group = Self.defaultWaveGroups.firstIndex {
                $0.contains(leg)
            }!
            var localStep = memory.gaitStep - group * configuration.swingSteps
            localStep %= cycleSteps
            if localStep < 0 { localStep += cycleSteps }

            let neutral = memory.neutralFeetInBody[leg]
            // Velocity of this foothold induced by the desired rigid-body
            // planar twist: v + omega x r. Stance applies the opposite foot
            // velocity, producing the requested body reaction through contact.
            var translation = SIMD2<Float>(command.x, command.y)
            let translationSpeed = simd_length(translation)
            if translationSpeed > configuration.standingCommandThreshold
                && translationSpeed < configuration.minimumTranslationSpeed {
                translation *= configuration.minimumTranslationSpeed
                    / translationSpeed
            }
            // A two-DOF leg cannot simultaneously realize the full lateral
            // and yaw components of a large twist. Turn first for distant
            // bearings, then blend translation in continuously as the target
            // enters the forward sector. This is the classical high-level
            // steering state machine that a learned policy gets implicitly.
            let translationBlend = simd_clamp(
                (0.70 - abs(command.z)) / 0.35, 0, 1)
            translation *= translationBlend
            var halfStride = SIMD2<Float>(
                translation.x - command.z * neutral.y,
                translation.y + command.z * neutral.x)
                * configuration.placementHorizon
            let strideLength = simd_length(halfStride)
            if strideLength > configuration.maximumPlanarPlacement {
                halfStride *= configuration.maximumPlanarPlacement
                    / strideLength
            }
            var footInBody = neutral
            if localStep < configuration.swingSteps {
                let phase = Float(localStep + 1)
                    / Float(configuration.swingSteps)
                let smooth = phase * phase * (3 - 2 * phase)
                footInBody.x += (2 * smooth - 1) * halfStride.x
                footInBody.y += (2 * smooth - 1) * halfStride.y
                footInBody.z += configuration.swingHeight * sin(.pi * phase)
                swingLeg = swingLeg ?? leg
                reportedSwingPhase = phase
            } else {
                let stanceSteps = cycleSteps - configuration.swingSteps
                let phase = Float(localStep - configuration.swingSteps + 1)
                    / Float(stanceSteps)
                footInBody.x += (1 - 2 * phase) * halfStride.x
                footInBody.y += (1 - 2 * phase) * halfStride.y
            }
            let solution = Self.inverseKinematics(
                leg: leg, footInBody: footInBody)
            if solution.wasConstrained {
                diagnostics.constrainedTargetCount += 1
            }
            actions[environment, 2 * leg] = simd_clamp(
                solution.hip / Arachne15PolicyContract.actionScales[2 * leg],
                -1, 1)
            actions[environment, 2 * leg + 1] = simd_clamp(
                solution.knee
                    / Arachne15PolicyContract.actionScales[2 * leg + 1],
                -1, 1)
        }
        if memory.moving {
            diagnostics.activeSwingLeg[environment] = swingLeg
            diagnostics.swingPhase[environment] = reportedSwingPhase
            memory.gaitStep = (memory.gaitStep + 1) % cycleSteps
        }
        environments[environment] = memory
    }

    public struct IKSolution: Sendable, Equatable {
        public var hip: Float
        public var knee: Float
        public var wasConstrained: Bool
    }

    /// Exact two-axis IK for the authored Arachne mechanism. Since the leg has
    /// yaw plus pitch rather than a planar two-link elbow, targets outside the
    /// tibia's reachable circle are projected by direction and reported.
    public static func inverseKinematics(
        leg: Int, footInBody target: F3
    ) -> IKSolution {
        precondition((0..<8).contains(leg))
        let hipPosition = hipPositions[leg]
        let delta = target - hipPosition
        let azimuth = atan2(delta.y, delta.x)
        let hipUnclamped = wrappedAngle(azimuth - fixedHipYaws[leg])
        let hip = simd_clamp(hipUnclamped, -0.35, 0.35)
        let direction = F3(cos(fixedHipYaws[leg] + hip),
                           sin(fixedHipYaws[leg] + hip), 0)
        let kneePivot = hipPosition + direction * coxaLength
        let tibiaTarget = target - kneePivot
        let radial = simd_dot(tibiaTarget, direction)
        let pitch = atan2(-tibiaTarget.z, radial)
        let kneeUnclamped = wrappedAngle(pitch - fixedTibiaPitch)
        let knee = simd_clamp(kneeUnclamped, -0.45, 0.45)
        let offPlane = tibiaTarget - direction * radial
            - F3(0, 0, tibiaTarget.z)
        let reachError = abs(sqrt(radial * radial
            + tibiaTarget.z * tibiaTarget.z) - tibiaLength)
        return IKSolution(
            hip: hip, knee: knee,
            wasConstrained: abs(hip - hipUnclamped) > 1e-5
                || abs(knee - kneeUnclamped) > 1e-5
                || simd_length(offPlane) > 0.004 || reachError > 0.015)
    }

    public static func forwardKinematics(
        leg: Int, hip: Float, knee: Float
    ) -> F3 {
        precondition((0..<8).contains(leg))
        let direction = F3(cos(fixedHipYaws[leg] + hip),
                           sin(fixedHipYaws[leg] + hip), 0)
        let pitch = fixedTibiaPitch + knee
        return hipPositions[leg]
            + direction * (coxaLength + tibiaLength * cos(pitch))
            + F3(0, 0, -tibiaLength * sin(pitch))
    }

    private static func wrappedAngle(_ angle: Float) -> Float {
        atan2(sin(angle), cos(angle))
    }
}
