import AVBDCore
import Foundation
import simd

public struct GEARSonicG1ReplayReport: Codable, Sendable {
    public var schemaVersion: Int
    public var referenceName: String
    public var referenceFrames: Int
    public var environmentCount: Int
    public var trajectoryEnvironmentIndex: Int
    public var controlSteps: Int
    public var simulatedSeconds: Float
    public var completedReference: Bool
    public var actualRootDisplacementMeters: [Float]
    public var referenceRootDisplacementMeters: [Float]
    public var meanJointTrackingErrorRadians: Float
    public var maximumJointTrackingErrorRadians: Float
    public var meanRootRelativeBodyTrackingErrorMeters: Float
    public var maximumRootRelativeBodyTrackingErrorMeters: Float
    public var maximumTrackedBodyHeightErrorMeters: Float
    public var maximumSourceCriterionHeightErrorMeters: Float
    public var maximumRootOrientationErrorRadians: Float
    public var maximumAbsoluteJointVelocityRadiansPerSecond: Float
    public var maximumJointVelocityLimitRatio: Float
    public var finalRootHeightMeters: Float
    public var minimumRootHeightMeters: Float
    public var minimumUprightAlignment: Float
    public var firstFallStep: Int?
    public var sourceHeightFailureThresholdMeters: Float
    public var sourceOrientationFailureThresholdRadians: Float
    public var firstSourceCriterionFailureStep: Int?
    public var sourceCriteriaPassed: Bool
    public var jointVelocityLimitsEnforced: Bool
    public var finite: Bool
    public var policyMaximumTokenError: Float
    public var policyMaximumActionError: Float
    public var policyParityPassed: Bool
}

/// Exact closed-loop GEAR-SONIC replay over the analytic G1 training plant.
///
/// Each control transition follows NVIDIA's released ordering: measure the
/// current state, append history carrying the previous raw action, construct
/// the current reference window, infer, hold the resulting target for four
/// 5-ms physics steps, then advance the 50-Hz reference cursor.
public final class GEARSonicG1Session {
    public let environment: GEARSonicG1Sim2SimEnv
    public let policy: GEARSonicG1Policy
    public let reference: GEARSonicG1ReferenceClip
    public let referenceName: String
    public let policyVerification: GEARSonicG1PolicyVerification

    public private(set) var referenceFrame = 0
    public private(set) var isPlaying = true
    public private(set) var completedReference = false
    public private(set) var controlSteps = 0
    public private(set) var elapsedTime: Float = 0
    public private(set) var previousRawAction: ContiguousArray<Float>
    public private(set) var lastReferenceObservation = ContiguousArray<Float>()
    public private(set) var lastHistoryObservation = ContiguousArray<Float>()
    public private(set) var lastRawAction = ContiguousArray<Float>()

    private var history: GEARSonicG1HistoryBuffer
    private var headingAlignments: [GEARSonicG1HeadingAlignment]
    private let initialRootPositions: [F3]
    private let initialReferenceRootPosition: F3
    private var jointErrorSum: Double = 0
    private var jointErrorCount = 0
    private var maximumJointError: Float = 0
    private var bodyTrackingErrorSum: Double = 0
    private var bodyTrackingErrorCount = 0
    private var maximumBodyTrackingError: Float = 0
    private var maximumTrackedBodyHeightError: Float = 0
    private var maximumSourceCriterionHeightError: Float = 0
    private var maximumRootOrientationError: Float = 0
    private var maximumAbsoluteJointVelocity: Float = 0
    private var maximumJointVelocityLimitRatio: Float = 0
    private var minimumHeight: Float
    private var minimumUpright: Float = 1
    private var firstFallStep: Int?
    private var firstSourceCriterionFailureStep: Int?
    private var allFinite = true

    // Released reference rows identify Isaac Lab body indices
    // [0,4,10,18,5,11,19,9,16,22,28,17,23,29]. Applying NVIDIA's pinned
    // IsaacLab-to-MuJoCo body permutation yields these plant indices.
    private static let trackedPlantBodyIndices = [
        0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29,
    ]
    // NVIDIA's evaluation termination checks pelvis height separately and
    // the two ankles plus two wrists through `ee_body_pos`.
    private static let sourceHeightCriterionBodyIndices = [0, 3, 6, 10, 13]
    private static let sourceHeightFailureThreshold: Float = 0.25
    private static let sourceOrientationFailureThreshold: Float = 1

    public init(
        bundleDirectory: String,
        referenceDirectory: String,
        environmentCount: Int = 1,
        solverIterations: Int = 8,
        includeVisuals: Bool = false,
        initializeFromReference: Bool = true
    ) throws {
        let bundle = URL(fileURLWithPath: bundleDirectory, isDirectory: true)
        let plantURL = bundle.appendingPathComponent("plant.xml")
        guard FileManager.default.fileExists(atPath: plantURL.path) else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC bundle is missing imported plant.xml")
        }
        policy = try GEARSonicG1Policy(directory: bundle.path)
        policyVerification = try policy.verifyGoldenBatch()
        guard policyVerification.passed else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC MLX policy failed its imported ONNX golden batch")
        }
        let loadedReference = try GEARSonicG1ReferenceClip(
            directory: referenceDirectory)
        reference = loadedReference
        referenceName = URL(fileURLWithPath: referenceDirectory,
                            isDirectory: true).lastPathComponent
        let control = policy.manifest.control
        environment = try GEARSonicG1Sim2SimEnv(
            plantURL: plantURL,
            configuration: .init(
                environmentCount: environmentCount,
                actuatorJointNames: control.actuatorJointNames,
                defaultJointPositions: control.defaultJointPositions,
                stiffness: control.stiffness,
                damping: control.damping,
                armature: control.trainingArmature,
                effortLimit: control.trainingEffortLimit,
                velocityLimit: control.trainingVelocityLimit,
                motorMode: .implicitPositionPD,
                physicsTimeStep: control.physicsTimeStep,
                controlDecimation: control.controlDecimation,
                rootHeight: 0.76,
                solverIterations: solverIterations,
                includeVisuals: includeVisuals))
        history = try GEARSonicG1HistoryBuffer(batchSize: environmentCount)
        previousRawAction = .init(
            repeating: 0, count: environmentCount * 29)

        if initializeFromReference {
            let policyQ = loadedReference.jointPositionsPolicyOrder[0]
            var actuatorQ = [Float](repeating: 0, count: 29)
            for actuator in 0..<29 {
                actuatorQ[actuator] = Float(
                    policyQ[control.actuatorToPolicy[actuator]])
            }
            let rootPosition = loadedReference.bodyPositions[0]
            let rootQuaternion = try loadedReference.rootQuaternionWXYZ(at: 0)
            let rootLinearVelocity = try loadedReference.rootLinearVelocity(at: 0)
            let rootAngularVelocity = try loadedReference.rootAngularVelocity(at: 0)
            let rotation = Self.simdQuaternion(rootQuaternion)
            let resets = environment.environmentOrigins.map { origin in
                GEARSonicG1Sim2SimEnv.ResetState(
                    sourceJointPositions: actuatorQ,
                    rootPosition: F3(
                        origin.x + Float(rootPosition[0]),
                        origin.y + Float(rootPosition[1]),
                        Float(rootPosition[2])),
                    rootRotation: rotation,
                    rootLinearVelocity: Self.simdVector(rootLinearVelocity),
                    rootAngularVelocity: Self.simdVector(rootAngularVelocity))
            }
            try environment.reset(resets)
        }

        let states = environment.states()
        var initialAlignments: [GEARSonicG1HeadingAlignment] = []
        initialAlignments.reserveCapacity(states.count)
        for state in states {
            initialAlignments.append(try loadedReference.headingAlignment(
                robotInitialBaseQuaternionWXYZ:
                    Self.referenceQuaternion(state.root.rotation),
                referenceFrame: 0))
        }
        headingAlignments = initialAlignments
        initialRootPositions = states.map(\.root.position)
        let p = loadedReference.bodyPositions[0]
        initialReferenceRootPosition = F3(
            Float(p[0]), Float(p[1]), Float(p[2]))
        minimumHeight = states.map(\.root.position.z).min() ?? .infinity
    }

    @discardableResult
    public func step() throws -> [HumanoidState] {
        guard !completedReference else { return environment.states() }
        let control = policy.manifest.control
        let measured = environment.states()
        try appendHistory(states: measured, control: control)
        let historyObservation = history.observations()
        var referenceObservation = ContiguousArray<Float>()
        referenceObservation.reserveCapacity(measured.count * 640)
        for environment in measured.indices {
            let values = try reference.referenceObservation640(
                currentFrame: referenceFrame,
                isPlaying: isPlaying,
                robotBaseQuaternionWXYZ:
                    Self.referenceQuaternion(measured[environment].root.rotation),
                headingAlignment: headingAlignments[environment])
            referenceObservation.append(contentsOf: values.map(Float.init))
        }
        let action = try policy.actions(
            referenceObservations: referenceObservation,
            historyObservations: historyObservation)
        let targets = try GEARSonicG1Control.jointPositionTargets(
            rawPolicyActions: action, control: control)
        try environment.step(sourceJointPositionTargets: targets)

        lastReferenceObservation = referenceObservation
        lastHistoryObservation = historyObservation
        lastRawAction = action
        previousRawAction = action
        let commandedFrame = referenceFrame
        controlSteps += 1
        elapsedTime += control.physicsTimeStep
            * Float(control.controlDecimation)
        if isPlaying {
            referenceFrame += 1
            if referenceFrame == reference.frameCount {
                // Match NVIDIA's non-planner end transition exactly.
                referenceFrame = 0
                isPlaying = false
                completedReference = true
            }
        }
        let result = environment.states()
        updateMetrics(
            states: result,
            commandedReferenceFrame: commandedFrame,
            bodyTrackingReferenceFrame: commandedFrame + 1
                < reference.frameCount ? commandedFrame + 1 : nil,
            control: control)
        return result
    }

    public func run(
        maximumControlSteps: Int? = nil,
        onStep: ((Int, [HumanoidState]) -> Void)? = nil
    ) throws -> GEARSonicG1ReplayReport {
        let maximum = maximumControlSteps ?? reference.frameCount
        precondition(maximum >= 0)
        while !completedReference && controlSteps < maximum {
            let state = try step()
            onStep?(controlSteps, state)
        }
        return report()
    }

    public func report() -> GEARSonicG1ReplayReport {
        let state = environment.states()[0]
        let actual = state.root.position - initialRootPositions[0]
        // Source evaluation records the post-transition state against the
        // advanced reference clock. At completion the last valid reference
        // remains the endpoint even though the deployment cursor wraps.
        let finalReferenceIndex = max(0, min(
            controlSteps, reference.frameCount - 1))
        let p = reference.bodyPositions[finalReferenceIndex]
        let rawReferenceDisplacement = F3(
            Float(p[0]), Float(p[1]), Float(p[2]))
            - initialReferenceRootPosition
        let referenceDisplacement = Self.simdQuaternion(
            headingAlignments[0].referenceAlignmentQuaternionWXYZ)
            .act(rawReferenceDisplacement)
        return GEARSonicG1ReplayReport(
            schemaVersion: 3,
            referenceName: referenceName,
            referenceFrames: reference.frameCount,
            environmentCount: environment.configuration.environmentCount,
            trajectoryEnvironmentIndex: 0,
            controlSteps: controlSteps,
            simulatedSeconds: elapsedTime,
            completedReference: completedReference,
            actualRootDisplacementMeters: [actual.x, actual.y, actual.z],
            referenceRootDisplacementMeters: [
                referenceDisplacement.x, referenceDisplacement.y,
                referenceDisplacement.z,
            ],
            meanJointTrackingErrorRadians: jointErrorCount > 0
                ? Float(jointErrorSum / Double(jointErrorCount)) : 0,
            maximumJointTrackingErrorRadians: maximumJointError,
            meanRootRelativeBodyTrackingErrorMeters:
                bodyTrackingErrorCount > 0
                    ? Float(bodyTrackingErrorSum
                        / Double(bodyTrackingErrorCount)) : 0,
            maximumRootRelativeBodyTrackingErrorMeters:
                maximumBodyTrackingError,
            maximumTrackedBodyHeightErrorMeters:
                maximumTrackedBodyHeightError,
            maximumSourceCriterionHeightErrorMeters:
                maximumSourceCriterionHeightError,
            maximumRootOrientationErrorRadians:
                maximumRootOrientationError,
            maximumAbsoluteJointVelocityRadiansPerSecond:
                maximumAbsoluteJointVelocity,
            maximumJointVelocityLimitRatio:
                maximumJointVelocityLimitRatio,
            finalRootHeightMeters: state.root.position.z,
            minimumRootHeightMeters: minimumHeight,
            minimumUprightAlignment: minimumUpright,
            firstFallStep: firstFallStep,
            sourceHeightFailureThresholdMeters:
                Self.sourceHeightFailureThreshold,
            sourceOrientationFailureThresholdRadians:
                Self.sourceOrientationFailureThreshold,
            firstSourceCriterionFailureStep:
                firstSourceCriterionFailureStep,
            sourceCriteriaPassed: firstSourceCriterionFailureStep == nil,
            // Isaac Lab's `velocity_limit_sim` is retained and audited here,
            // but AVBD does not yet clamp generalized velocity in the solve.
            jointVelocityLimitsEnforced: false,
            finite: allFinite,
            policyMaximumTokenError: policyVerification.maximumTokenError,
            policyMaximumActionError: policyVerification.maximumActionError,
            policyParityPassed: policyVerification.passed)
    }

    private func appendHistory(
        states: [HumanoidState],
        control: GEARSonicG1PolicyManifest.Control
    ) throws {
        var angularVelocity = ContiguousArray<Float>()
        var positionResidual = ContiguousArray<Float>()
        var jointVelocity = ContiguousArray<Float>()
        var projectedGravity = ContiguousArray<Float>()
        angularVelocity.reserveCapacity(states.count * 3)
        positionResidual.reserveCapacity(states.count * 29)
        jointVelocity.reserveCapacity(states.count * 29)
        projectedGravity.reserveCapacity(states.count * 3)
        for state in states {
            let localAngular = state.root.rotation.inverse.act(
                state.root.angularVelocity)
            let gravity = state.root.rotation.inverse.act(F3(0, 0, -1))
            angularVelocity.append(contentsOf: [
                localAngular.x, localAngular.y, localAngular.z,
            ])
            projectedGravity.append(contentsOf: [
                gravity.x, gravity.y, gravity.z,
            ])
            for policyJoint in 0..<29 {
                let actuator = control.policyToActuator[policyJoint]
                positionResidual.append(
                    state.jointAngles[actuator]
                        - control.defaultJointPositions[actuator])
                jointVelocity.append(state.jointVelocities[actuator])
            }
        }
        try history.append(
            baseAngularVelocity: angularVelocity,
            jointPositionResidual: positionResidual,
            jointVelocity: jointVelocity,
            previousRawAction: previousRawAction,
            projectedGravity: projectedGravity)
    }

    private func updateMetrics(
        states: [HumanoidState], commandedReferenceFrame: Int,
        bodyTrackingReferenceFrame: Int?,
        control: GEARSonicG1PolicyManifest.Control
    ) {
        let referenceQ = reference.jointPositionsPolicyOrder[
            commandedReferenceFrame]
        let trackedStates = environment.linkStates(
            bodyIndices: Self.trackedPlantBodyIndices)
        let sourceReferencePositions = reference.bodyPositions[
            commandedReferenceFrame]
        let referenceRootQuaternion = try? reference.rootQuaternionWXYZ(
            at: commandedReferenceFrame)
        let trackingReferencePositions = bodyTrackingReferenceFrame.map {
            reference.bodyPositions[$0]
        }
        for environmentIndex in states.indices {
            let state = states[environmentIndex]
            let upright = state.root.rotation.act(F3(0, 0, 1)).z
            minimumHeight = min(minimumHeight, state.root.position.z)
            minimumUpright = min(minimumUpright, upright)
            if firstFallStep == nil
                && (state.root.position.z < 0.35 || upright < 0.2) {
                firstFallStep = controlSteps
            }
            for policyJoint in 0..<29 {
                let actuator = control.policyToActuator[policyJoint]
                let error = abs(state.jointAngles[actuator]
                    - Float(referenceQ[policyJoint]))
                jointErrorSum += Double(error)
                jointErrorCount += 1
                maximumJointError = max(maximumJointError, error)
            }
            for actuator in 0..<29 {
                let speed = abs(state.jointVelocities[actuator])
                maximumAbsoluteJointVelocity = max(
                    maximumAbsoluteJointVelocity, speed)
                maximumJointVelocityLimitRatio = max(
                    maximumJointVelocityLimitRatio,
                    speed / control.trainingVelocityLimit[actuator])
            }

            let links = trackedStates[environmentIndex]
            let alignment = Self.simdQuaternion(
                headingAlignments[environmentIndex]
                    .referenceAlignmentQuaternionWXYZ)
            var sourceFrameMaximumHeightError: Float = 0
            for body in Self.sourceHeightCriterionBodyIndices {
                let referenceZ = Float(sourceReferencePositions[body * 3 + 2])
                sourceFrameMaximumHeightError = max(
                    sourceFrameMaximumHeightError,
                    abs(links[body].position.z - referenceZ))
            }
            maximumSourceCriterionHeightError = max(
                maximumSourceCriterionHeightError,
                sourceFrameMaximumHeightError)

            if let trackingReferencePositions {
                let actualRootPosition = links[0].position
                let referenceRootPosition = F3(
                    Float(trackingReferencePositions[0]),
                    Float(trackingReferencePositions[1]),
                    Float(trackingReferencePositions[2]))
                for body in links.indices {
                    let offset = body * 3
                    let referencePosition = F3(
                        Float(trackingReferencePositions[offset]),
                        Float(trackingReferencePositions[offset + 1]),
                        Float(trackingReferencePositions[offset + 2]))
                    let expectedRelative = alignment.act(
                        referencePosition - referenceRootPosition)
                    let actualRelative = links[body].position
                        - actualRootPosition
                    let error = simd_length(actualRelative - expectedRelative)
                    bodyTrackingErrorSum += Double(error)
                    bodyTrackingErrorCount += 1
                    maximumBodyTrackingError = max(
                        maximumBodyTrackingError, error)
                    maximumTrackedBodyHeightError = max(
                        maximumTrackedBodyHeightError,
                        abs(links[body].position.z - referencePosition.z))
                }
            }
            var rootOrientationError: Float = .infinity
            if let referenceRootQuaternion {
                let expectedRoot = (alignment
                    * Self.simdQuaternion(referenceRootQuaternion)).normalized
                let difference = (state.root.rotation.inverse
                    * expectedRoot).normalized
                rootOrientationError = 2 * acos(min(
                    max(abs(difference.real), 0), 1))
                maximumRootOrientationError = max(
                    maximumRootOrientationError, rootOrientationError)
            } else {
                allFinite = false
            }
            if firstSourceCriterionFailureStep == nil
                && (sourceFrameMaximumHeightError
                        > Self.sourceHeightFailureThreshold
                    || rootOrientationError
                        > Self.sourceOrientationFailureThreshold) {
                firstSourceCriterionFailureStep = controlSteps
            }
            let finite = [
                state.root.position.x, state.root.position.y,
                state.root.position.z, upright, rootOrientationError,
            ] + state.jointAngles + state.jointVelocities
            if !finite.allSatisfy(\.isFinite) { allFinite = false }
            for link in links {
                let linkValues = [
                    link.position.x, link.position.y, link.position.z,
                    link.rotation.real, link.rotation.imag.x,
                    link.rotation.imag.y, link.rotation.imag.z,
                    link.linearVelocity.x, link.linearVelocity.y,
                    link.linearVelocity.z, link.angularVelocity.x,
                    link.angularVelocity.y, link.angularVelocity.z,
                ]
                if !linkValues.allSatisfy(\.isFinite) { allFinite = false }
            }
        }
    }

    private static func referenceQuaternion(
        _ quaternion: Quat
    ) -> GEARSonicWXYZQuaternion {
        GEARSonicWXYZQuaternion(
            w: Double(quaternion.real),
            x: Double(quaternion.imag.x),
            y: Double(quaternion.imag.y),
            z: Double(quaternion.imag.z))
    }

    private static func simdQuaternion(
        _ quaternion: GEARSonicWXYZQuaternion
    ) -> Quat {
        Quat(real: Float(quaternion.w), imag: F3(
            Float(quaternion.x), Float(quaternion.y), Float(quaternion.z)))
            .normalized
    }

    private static func simdVector(
        _ vector: GEARSonicXYZVector?
    ) -> F3 {
        guard let vector else { return .zero }
        return F3(Float(vector.x), Float(vector.y), Float(vector.z))
    }
}
