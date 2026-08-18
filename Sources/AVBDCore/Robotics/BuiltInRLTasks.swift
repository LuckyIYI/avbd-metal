/// Canonical task registry used by the CLI and by MLX experiments. A new
/// scene/task becomes trainable by registering one factory; algorithms do not
/// require task-specific branches.
public enum BuiltInRLTasks {
    private static let pushTOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "actionScale",
    ]
    private static let humanoidWalkOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "minimumCommandSpeed",
        "maximumCommandSpeed", "commandCurriculumControlSteps",
        "trainingInitialEpisodeAgeFraction", "standingCommandProbability",
        "standingCommandCurriculumControlSteps", "commandGatedActor",
        "threeModeActor", "expertGateCommandSpeed", "expertGateBlendWidth",
        "standExpertBlendStartSpeed", "standExpertBlendWidth",
        "standExpertRequiresDoubleSupport",
        "standExpertUsesPlanarSpeed",
        "freezeBasePolicyExpert", "freezeLowSpeedPolicyExpert",
        "trainBasePolicyExpert", "velocityTrackingStandardDeviation",
        "velocityTrackingErrorPenaltyWeight", "standStillVelocityPenaltyWeight",
        "standStillJointDeviationPenaltyWeight",
        "standStillDoubleSupportRewardWeight", "standStillFallPenalty",
        "lateralPenaltyWarmupControlSteps", "lateralPenaltyRampControlSteps",
        "laneTrackingStandardDeviation", "alternatingTouchdownRewardWeight",
        "flightPenaltyWeight", "initialRollPitchRange", "initialYawRange",
        "actionTargetResponse",
    ]
    private static let humanoidVelocityOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation",
        "minimumForwardVelocity", "maximumForwardVelocity",
        "minimumLateralVelocity", "maximumLateralVelocity",
        "minimumYawRate", "maximumYawRate", "commandResamplingSteps",
        "standingCommandProbability", "initialRollPitchRange",
        "initialYawRange",
    ]
    private static let humanoidIsaacVelocityOptionKeys: Set<String> = [
        "maxEpisodeSteps", "commandResamplingSteps",
        "standingCommandProbability", "initialYawRange",
        "observationNoise", "solverIterations",
    ]
    private static let humanoidIsaacGoalOptionKeys =
        humanoidIsaacVelocityOptionKeys.union([
            "pointGoal",
            "minimumGoalDistance", "maximumGoalDistance", "goalRadius",
            "goalSlowdownDistance", "goalCommandSpeed",
            "goalBoundaryCommandSpeed", "goalDwellSteps",
            "maximumGoalArrivalSpeed", "goalProgressRewardWeight",
            "goalStableRewardWeight", "goalSuccessBonus",
            "projectileProbability", "projectileCurriculumControlSteps",
            "projectileSize", "projectileMass",
            "minimumProjectileSpeed", "maximumProjectileSpeed",
            "projectileLeftProbability",
            "minimumProjectileLaunchStep", "maximumProjectileLaunchStep",
            "recoveryGatedActor", "freezeBasePolicyExpert",
            "recoveryContextObservations", "recoveryContextDuration",
            "recoveryExpertSide", "recoveryExpertGatePeak",
            "recoveryExpertGateDecay",
            "initializeRecoveryExpertFromBaseOnTransfer",
            "initializeRecoveryExpertFromMirroredBaseOnTransfer",
            "postImpactUprightRewardWeight",
            "postImpactAngularVelocityPenaltyWeight",
            "postImpactFallPenalty",
        ])
    private static let humanoidBoxCarryOptionKeys: Set<String> = [
        "maxEpisodeSteps", "solverIterations", "observationNoise",
        "minimumTrainingStationDistance", "evaluationStationDistance",
        "stationDistanceCurriculumControlSteps", "boxMass", "boxFriction",
        "minimumTrainingLiftClearance", "liftClearance",
        "liftClearanceCurriculumControlSteps",
        "minimumTrainingCarryDistance", "carryDistance",
        "carryDistanceCurriculumControlSteps",
        "destinationBearingCurriculumControlSteps",
        "minimumTrainingSuccessDwellSteps", "successDwellSteps",
        "successDwellCurriculumControlSteps", "pregraspForwardOffset",
        "pregraspLateralOffset", "approachCommandSpeed",
        "carryCommandSpeed", "carryHolonomicCommand",
        "navigationGoalObservationScale",
        "manipulationHandoffSteps",
        "carryHandoffSteps", "carryCommandRampSteps",
        "manipulationArmActionScaleMultiplier",
        "carryBaseLegActionFraction",
        "freezeBasePolicyExpert",
        "freezeManipulationPolicyExpert",
        "freezeCarryPolicyExpert",
        "manipulationGatedActor",
        "initializeManipulationExpertFromBaseOnTransfer",
        "initializeCarryExpertFromManipulationExpertOnTransfer",
        "compositionalCarryController",
        "upperBodyCarryController",
        "carryLocomotionControlsTorso",
        "initializeCarryExpertFromBaseOnTransfer",
        "initializeCarryLocomotionExpertFromBaseOnTransfer",
        "carryStartReplayProbability",
        "advanceReplaySnapshotAtDestinationContact",
        "carryArmReferenceWeight",
        "carryHoldClearanceMultiplier",
        "carryProgressRewardWeight",
        "carryLocomotionRewardMultiplier",
        "carryTrackingVariance",
        "coupledCarryCommandTracking",
        "carryRootProgressRewardWeight",
        "carryAlternatingStepRewardWeight",
        "minimumLoadedAlternatingSteps",
    ]
    private static let humanoidGoalOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "minimumCommandSpeed",
        "maximumCommandSpeed", "commandCurriculumControlSteps",
        "trainingInitialEpisodeAgeFraction", "standingCommandProbability",
        "commandGatedActor",
        "threeModeActor", "expertGateCommandSpeed", "expertGateBlendWidth",
        "standExpertBlendStartSpeed", "standExpertBlendWidth",
        "standExpertRequiresDoubleSupport",
        "standExpertUsesPlanarSpeed",
        "freezeBasePolicyExpert", "freezeLowSpeedPolicyExpert",
        "trainBasePolicyExpert", "velocityTrackingStandardDeviation",
        "velocityTrackingErrorPenaltyWeight", "standStillVelocityPenaltyWeight",
        "standStillJointDeviationPenaltyWeight",
        "standStillDoubleSupportRewardWeight", "standStillFallPenalty",
        "lateralPenaltyWarmupControlSteps", "lateralPenaltyRampControlSteps",
        "laneTrackingStandardDeviation", "alternatingTouchdownRewardWeight",
        "flightPenaltyWeight", "initialRollPitchRange", "initialYawRange",
        "maximumGoalDirectionAngle", "initialGoalDirectionAngle",
        "goalDirectionCurriculumControlSteps", "goalRadius",
        "goalSlowdownDistance", "goalObservationUsesLateralVelocity",
        "goalObservationIncludesLateralVelocity",
        "minimumGoalDistanceMeters",
        "maximumGoalDistanceMeters", "initialGoalDistanceScale",
        "goalDistanceCurriculumControlSteps", "goalDwellSteps",
        "maximumGoalArrivalSpeed", "goalBoundaryCommandSpeed",
        "goalStableDwellRewardWeight", "projectileProbability",
        "projectileCurriculumControlSteps", "minimumProjectileSpeed",
        "maximumProjectileSpeed", "minimumProjectileLaunchStep",
        "maximumProjectileLaunchStep", "actionTargetResponse",
    ]
    private static let armPushTOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "jointDeltaActionScale",
        "endEffectorDeltaActionScale",
        "linkLength1", "linkLength2",
        "linkMass", "tipMass", "motorTorque", "motorStiffness",
        "motorDamping", "motorArmature", "blockMass",
        "blockStaticFriction", "blockDynamicFriction",
        "blockSpawnGoalBlend",
        "blockSpawnRadius", "blockSpawnYawRange", "blockSpawnLateralBias",
        "goalProgressWeight",
        "reachProgressWeight", "yawProgressWeight", "coverageProgressWeight",
        "coverageRewardWeight", "poseProgressRewardWeight", "poseRewardWeight",
        "poseRewardDistanceScale",
        "reachingRewardWeight", "actionMagnitudePenaltyWeight",
        "actionRatePenaltyWeight", "precisionGatedActor",
        "precisionExpertGateCoverage", "precisionExpertReleaseCoverage",
        "freezeBasePolicyExpert",
        "pushContactProgressWeight",
        "reachDistancePenaltyWeight", "goalDistancePenaltyWeight",
        "yawErrorPenaltyWeight", "pushContactDistancePenaltyWeight",
        "successBonus", "successRewardOverride", "successCoverage",
        "continueAfterSuccess",
        "successYawTolerance",
        "reachCurriculumSuccessDistance",
        "pushContactCurriculumSuccessDistance", "pushContactOffset",
        "pushContactCurriculumMaximumGoalRegression",
        "pushContactCurriculumMinimumGoalProgress",
        "pushContactCurriculumMaximumGoalDistance",
    ]
    private static let arachne15OptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "solverIterations",
        "commandResamplingSteps",
        "minimumForwardVelocity", "maximumForwardVelocity",
        "maximumLateralVelocity", "maximumYawRate",
        "standingCommandProbability", "initialRollPitchRange",
        "initialYawRange", "observationNoise",
        "maximumActionLatencySteps", "validationCollisionProfile",
        "domainRandomization",
        "pointGoal", "minimumGoalDistance", "maximumGoalDistance",
        "maximumGoalDirectionAngle",
        "goalRadius", "goalSlowdownDistance", "goalCommandSpeed",
        "goalBoundaryCommandSpeed", "maximumGoalArrivalSpeed",
        "goalDwellSteps", "goalProgressRewardWeight",
        "goalStableRewardWeight", "goalSuccessBonus",
        "commandProgressRewardWeight", "velocityErrorPenaltyWeight",
        "yawErrorPenaltyWeight",
        "massScaleLower", "massScaleUpper",
        "inertiaScaleLower", "inertiaScaleUpper",
        "frictionScaleLower", "frictionScaleUpper",
        "motorTorqueScaleLower", "motorTorqueScaleUpper",
        "motorStiffnessScaleLower", "motorStiffnessScaleUpper",
        "motorDampingScaleLower", "motorDampingScaleUpper",
        "armatureScaleLower", "armatureScaleUpper",
    ]

    private static func arachne15Configuration(
        _ cfg: RLTaskConfiguration, pointGoal: Bool
    ) throws -> Arachne15LocomotionTaskConfig {
        if let serializedMode = cfg.options["pointGoal"],
           (serializedMode > 0) != pointGoal {
            throw RLEnvironmentError.invalidConfiguration(
                "Arachne pointGoal option disagrees with selected task")
        }
        let fallbackRandomization =
            (cfg.options["domainRandomization"] ?? 1) > 0
                ? ArticulationDomainRandomization.conservativeSimToReal
                : .init()
        func range(_ prefix: String,
                   _ fallback: DynamicsMultiplierRange)
            throws -> DynamicsMultiplierRange {
            let lower = cfg.options["\(prefix)Lower"] ?? fallback.lowerBound
            let upper = cfg.options["\(prefix)Upper"] ?? fallback.upperBound
            guard lower.isFinite, upper.isFinite,
                  lower > 0, upper >= lower else {
                throw RLEnvironmentError.invalidConfiguration(
                    "invalid Arachne \(prefix) range \(lower)...\(upper)")
            }
            return DynamicsMultiplierRange(lower, upper)
        }
        let randomization = ArticulationDomainRandomization(
            mass: try range("massScale", fallbackRandomization.mass),
            inertia: try range("inertiaScale", fallbackRandomization.inertia),
            friction: try range(
                "frictionScale", fallbackRandomization.friction),
            motorTorque: try range(
                "motorTorqueScale", fallbackRandomization.motorTorque),
            motorStiffness: try range(
                "motorStiffnessScale", fallbackRandomization.motorStiffness),
            motorDamping: try range(
                "motorDampingScale", fallbackRandomization.motorDamping),
            armature: try range(
                "armatureScale", fallbackRandomization.armature))
        let requestedGoalDirectionAngle =
            cfg.options["maximumGoalDirectionAngle"] ?? .pi
        guard requestedGoalDirectionAngle > 0,
              requestedGoalDirectionAngle <= Float.pi + 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "Arachne maximumGoalDirectionAngle must be in (0, pi]")
        }
        // Decimal CLI spellings of pi commonly round one Float ULP above
        // `Float.pi`. Canonicalizing that tolerance keeps checkpoint metadata
        // exact without accepting a genuinely wider-than-circle range.
        let maximumGoalDirectionAngle = min(
            requestedGoalDirectionAngle, Float.pi)
        return Arachne15LocomotionTaskConfig(
            numEnvironments: cfg.numEnvironments,
            seed: cfg.seed,
            maxEpisodeSteps: Int(cfg.options["maxEpisodeSteps"] ?? 1_000),
            controlDecimation: Int(cfg.options["controlDecimation"] ?? 10),
            solverIterations: Int(cfg.options["solverIterations"] ?? 20),
            commandResamplingSteps: Int(
                cfg.options["commandResamplingSteps"] ?? 500),
            minimumForwardVelocity:
                cfg.options["minimumForwardVelocity"] ?? 0.05,
            maximumForwardVelocity:
                cfg.options["maximumForwardVelocity"] ?? 0.25,
            maximumLateralVelocity:
                cfg.options["maximumLateralVelocity"] ?? 0.12,
            maximumYawRate: cfg.options["maximumYawRate"] ?? 0.8,
            standingCommandProbability:
                cfg.options["standingCommandProbability"]
                    ?? (pointGoal ? 0 : 0.10),
            initialRollPitchRange:
                cfg.options["initialRollPitchRange"] ?? 0.02,
            initialYawRange: cfg.options["initialYawRange"] ?? .pi,
            observationNoise: (cfg.options["observationNoise"] ?? 1) > 0,
            maximumActionLatencySteps: Int(
                cfg.options["maximumActionLatencySteps"] ?? 2),
            collisionProfile:
                (cfg.options["validationCollisionProfile"] ?? 0) > 0
                    ? .validation : .training,
            domainRandomization: randomization,
            pointGoal: pointGoal,
            minimumGoalDistance:
                cfg.options["minimumGoalDistance"] ?? 0.60,
            maximumGoalDistance:
                cfg.options["maximumGoalDistance"] ?? 2.0,
            maximumGoalDirectionAngle:
                maximumGoalDirectionAngle,
            goalRadius: cfg.options["goalRadius"] ?? 0.12,
            goalSlowdownDistance:
                cfg.options["goalSlowdownDistance"] ?? 0.50,
            goalCommandSpeed: cfg.options["goalCommandSpeed"] ?? 0.15,
            goalBoundaryCommandSpeed:
                cfg.options["goalBoundaryCommandSpeed"] ?? 0.02,
            maximumGoalArrivalSpeed:
                cfg.options["maximumGoalArrivalSpeed"] ?? 0.08,
            goalDwellSteps: Int(cfg.options["goalDwellSteps"] ?? 15),
            goalProgressRewardWeight:
                cfg.options["goalProgressRewardWeight"] ?? 4.0,
            goalStableRewardWeight:
                cfg.options["goalStableRewardWeight"] ?? 1.0,
            goalSuccessBonus: cfg.options["goalSuccessBonus"] ?? 8.0,
            commandProgressRewardWeight:
                cfg.options["commandProgressRewardWeight"] ?? 20.0,
            velocityErrorPenaltyWeight:
                cfg.options["velocityErrorPenaltyWeight"] ?? 5.0,
            yawErrorPenaltyWeight:
                cfg.options["yawErrorPenaltyWeight"] ?? 5.0,
            autoReset: cfg.autoReset)
    }
    private static let maniSkillPushTOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "jointDeltaActionScale",
        "robotInitialJointNoise", "normalizedDenseReward",
    ]

    /// Scalar experiment files stay portable while this schema restores the
    /// option types before any task factory can truncate or coerce a value.
    private static let integerOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "solverIterations",
        "commandResamplingSteps", "commandCurriculumControlSteps",
        "standingCommandCurriculumControlSteps",
        "lateralPenaltyWarmupControlSteps", "lateralPenaltyRampControlSteps",
        "goalDirectionCurriculumControlSteps",
        "goalDistanceCurriculumControlSteps", "goalDwellSteps",
        "projectileCurriculumControlSteps", "minimumProjectileLaunchStep",
        "maximumProjectileLaunchStep", "maximumActionLatencySteps",
        "stationDistanceCurriculumControlSteps",
        "carryDistanceCurriculumControlSteps",
        "destinationBearingCurriculumControlSteps",
        "minimumTrainingSuccessDwellSteps", "successDwellSteps",
        "successDwellCurriculumControlSteps",
    ]
    private static let positiveIntegerOptionKeys: Set<String> = [
        "maxEpisodeSteps", "controlDecimation", "solverIterations",
        "commandResamplingSteps", "goalDwellSteps",
        "minimumTrainingSuccessDwellSteps", "successDwellSteps",
    ]
    private static let booleanOptionKeys: Set<String> = [
        "commandGatedActor", "threeModeActor",
        "manipulationGatedActor",
        "standExpertRequiresDoubleSupport", "standExpertUsesPlanarSpeed",
        "freezeBasePolicyExpert", "freezeLowSpeedPolicyExpert",
        "trainBasePolicyExpert", "pointGoal", "observationNoise",
        "recoveryGatedActor", "recoveryContextObservations",
        "recoveryExpertGateDecay",
        "initializeRecoveryExpertFromBaseOnTransfer",
        "initializeRecoveryExpertFromMirroredBaseOnTransfer",
        "initializeManipulationExpertFromBaseOnTransfer",
        "freezeManipulationPolicyExpert",
        "freezeCarryPolicyExpert",
        "initializeCarryExpertFromManipulationExpertOnTransfer",
        "compositionalCarryController", "upperBodyCarryController",
        "initializeCarryExpertFromBaseOnTransfer",
        "carryHolonomicCommand",
        "carryLocomotionControlsTorso",
        "advanceReplaySnapshotAtDestinationContact",
        "goalObservationUsesLateralVelocity",
        "goalObservationIncludesLateralVelocity", "precisionGatedActor",
        "continueAfterSuccess", "validationCollisionProfile",
        "domainRandomization", "normalizedDenseReward",
    ]

    private static func optionSchema(
        _ optionNames: Set<String>
    ) -> RLTaskOptionSchema {
        RLTaskOptionSchema(optionNames.sorted().map { name in
            if booleanOptionKeys.contains(name) {
                return RLTaskOptionDefinition(name, valueKind: .boolean)
            }
            if integerOptionKeys.contains(name) {
                return RLTaskOptionDefinition(
                    name,
                    valueKind: .integer,
                    lowerBound: positiveIntegerOptionKeys.contains(name) ? 1 : 0)
            }
            return RLTaskOptionDefinition(name)
        })
    }

    public static let registry: RLTaskRegistry = {
        let registry = RLTaskRegistry()
        try! registry.register(
            "pusht-state-v0", optionSchema: optionSchema(pushTOptionKeys)
        ) { cfg in
            return try PushTTask(configuration: PushTTaskConfig(
                numEnvironments: cfg.numEnvironments,
                seed: cfg.seed,
                maxEpisodeSteps: Int(cfg.options["maxEpisodeSteps"] ?? 300),
                controlDecimation: Int(cfg.options["controlDecimation"] ?? 4),
                actionScale: cfg.options["actionScale"] ?? 0.18,
                autoReset: cfg.autoReset))
        }
        try! registry.register(
            "humanoid-walk-v0",
            optionSchema: optionSchema(humanoidWalkOptionKeys)
        ) { cfg in
            return try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
                numEnvironments: cfg.numEnvironments,
                seed: cfg.seed,
                maxEpisodeSteps: Int(cfg.options["maxEpisodeSteps"] ?? 1_000),
                controlDecimation: Int(cfg.options["controlDecimation"] ?? 4),
                minimumCommandSpeed: cfg.options["minimumCommandSpeed"] ?? 0.45,
                maximumCommandSpeed: cfg.options["maximumCommandSpeed"] ?? 0.65,
                commandCurriculumControlSteps:
                    Int(cfg.options["commandCurriculumControlSteps"] ?? 0),
                trainingInitialEpisodeAgeFraction:
                    cfg.options["trainingInitialEpisodeAgeFraction"] ?? 0,
                standingCommandProbability:
                    cfg.options["standingCommandProbability"] ?? 0,
                standingCommandCurriculumControlSteps: Int(
                    cfg.options["standingCommandCurriculumControlSteps"] ?? 0),
                commandGatedActor: (cfg.options["commandGatedActor"] ?? 0) > 0,
                threeModeActor: (cfg.options["threeModeActor"] ?? 0) > 0,
                expertGateCommandSpeed:
                    cfg.options["expertGateCommandSpeed"] ?? 0.20,
                expertGateBlendWidth:
                    cfg.options["expertGateBlendWidth"] ?? 0,
                standExpertBlendStartSpeed:
                    cfg.options["standExpertBlendStartSpeed"] ?? 0,
                standExpertBlendWidth:
                    cfg.options["standExpertBlendWidth"] ?? 0,
                standExpertRequiresDoubleSupport:
                    (cfg.options["standExpertRequiresDoubleSupport"] ?? 0) > 0,
                standExpertUsesPlanarSpeed:
                    (cfg.options["standExpertUsesPlanarSpeed"] ?? 0) > 0,
                freezeBasePolicyExpert:
                    (cfg.options["freezeBasePolicyExpert"] ?? 0) > 0,
                freezeLowSpeedPolicyExpert:
                    (cfg.options["freezeLowSpeedPolicyExpert"] ?? 0) > 0,
                trainBasePolicyExpert:
                    (cfg.options["trainBasePolicyExpert"] ?? 0) > 0,
                velocityTrackingStandardDeviation:
                    cfg.options["velocityTrackingStandardDeviation"]
                        ?? HumanoidLocomotionObjective
                            .velocityTrackingStandardDeviation,
                velocityTrackingErrorPenaltyWeight:
                    cfg.options["velocityTrackingErrorPenaltyWeight"] ?? 0,
                standStillVelocityPenaltyWeight:
                    cfg.options["standStillVelocityPenaltyWeight"] ?? 2,
                standStillJointDeviationPenaltyWeight:
                    cfg.options["standStillJointDeviationPenaltyWeight"] ?? 1,
                standStillDoubleSupportRewardWeight:
                    cfg.options["standStillDoubleSupportRewardWeight"] ?? 1,
                standStillFallPenalty:
                    cfg.options["standStillFallPenalty"] ?? 0,
                lateralPenaltyWarmupControlSteps:
                    Int(cfg.options["lateralPenaltyWarmupControlSteps"] ?? 0),
                lateralPenaltyRampControlSteps:
                    Int(cfg.options["lateralPenaltyRampControlSteps"] ?? 0),
                laneTrackingStandardDeviation:
                    cfg.options["laneTrackingStandardDeviation"] ?? 0.30,
                alternatingTouchdownRewardWeight:
                    cfg.options["alternatingTouchdownRewardWeight"] ?? 2,
                flightPenaltyWeight:
                    cfg.options["flightPenaltyWeight"] ?? 1,
                initialRollPitchRange:
                    cfg.options["initialRollPitchRange"] ?? 0.015,
                initialYawRange: cfg.options["initialYawRange"] ?? 0.05,
                actionTargetResponse: cfg.options["actionTargetResponse"] ?? 1,
                autoReset: cfg.autoReset),
                taskRevision: ((cfg.options["minimumCommandSpeed"] ?? 0.45) < 0.20
                    || (cfg.options["standingCommandProbability"] ?? 0) > 0
                    || (cfg.options["commandGatedActor"] ?? 0) > 0
                    ? ((cfg.options["standStillFallPenalty"] ?? 0) > 0
                        ? 45
                        : (cfg.options["commandGatedActor"] ?? 0) > 0
                        ? ((cfg.options["expertGateCommandSpeed"] ?? 0.20) > 0.20
                            ? 44 : 43)
                        : ((cfg.options["standingCommandProbability"] ?? 0) > 0
                            ? 42 : 41))
                    : 35)
                    + ((cfg.options["trainingInitialEpisodeAgeFraction"] ?? 0) > 0
                        ? 100 : 0)
                    + ((cfg.options["trainBasePolicyExpert"] ?? 0) > 0
                        ? 200 : 0)
                    + ((cfg.options["threeModeActor"] ?? 0) > 0
                        ? 400 : 0)
                    + ((cfg.options["velocityTrackingStandardDeviation"]
                        ?? HumanoidLocomotionObjective
                            .velocityTrackingStandardDeviation)
                        != HumanoidLocomotionObjective
                            .velocityTrackingStandardDeviation ? 800 : 0)
                    + ((cfg.options["velocityTrackingErrorPenaltyWeight"] ?? 0) > 0
                        ? 1_600 : 0)
                    + ((cfg.options["freezeLowSpeedPolicyExpert"] ?? 0) > 0
                        ? 3_200 : 0))
        }
        try! registry.register(
            "humanoid-velocity-v0",
            optionSchema: optionSchema(humanoidVelocityOptionKeys)
        ) { cfg in
            return try HumanoidVelocityTask(
                configuration: HumanoidVelocityTaskConfig(
                    numEnvironments: cfg.numEnvironments,
                    seed: cfg.seed,
                    maxEpisodeSteps: Int(
                        cfg.options["maxEpisodeSteps"] ?? 1_000),
                    controlDecimation: Int(
                        cfg.options["controlDecimation"] ?? 5),
                    minimumForwardVelocity:
                        cfg.options["minimumForwardVelocity"] ?? -0.6,
                    maximumForwardVelocity:
                        cfg.options["maximumForwardVelocity"] ?? 1.5,
                    minimumLateralVelocity:
                        cfg.options["minimumLateralVelocity"] ?? -0.8,
                    maximumLateralVelocity:
                        cfg.options["maximumLateralVelocity"] ?? 0.8,
                    minimumYawRate:
                        cfg.options["minimumYawRate"] ?? -0.7,
                    maximumYawRate:
                        cfg.options["maximumYawRate"] ?? 0.7,
                    commandResamplingSteps: Int(
                        cfg.options["commandResamplingSteps"] ?? 500),
                    standingCommandProbability:
                        cfg.options["standingCommandProbability"] ?? 0.10,
                    initialRollPitchRange:
                        cfg.options["initialRollPitchRange"] ?? 0.015,
                    initialYawRange:
                        cfg.options["initialYawRange"] ?? .pi,
                    autoReset: cfg.autoReset))
        }
        try! registry.register(
            "humanoid-isaac-flat-v0",
            optionSchema: optionSchema(humanoidIsaacVelocityOptionKeys)
        ) { cfg in
            return try HumanoidIsaacVelocityTask(
                configuration: HumanoidIsaacVelocityTaskConfig(
                    numEnvironments: cfg.numEnvironments,
                    seed: cfg.seed,
                    maxEpisodeSteps: Int(
                        cfg.options["maxEpisodeSteps"] ?? 1_000),
                    commandResamplingSteps: Int(
                        cfg.options["commandResamplingSteps"] ?? 500),
                    standingCommandProbability:
                        cfg.options["standingCommandProbability"] ?? 0.02,
                    initialYawRange:
                        cfg.options["initialYawRange"] ?? .pi,
                    observationNoise:
                        (cfg.options["observationNoise"] ?? 1) > 0,
                    solverIterations:
                        Int(cfg.options["solverIterations"] ?? 20),
                    autoReset: cfg.autoReset),
                includeInteractiveRobustnessProbe:
                    cfg.includeInteractiveRobustnessProbes)
        }
        try! registry.register(
            "humanoid-box-carry-v0",
            optionSchema: optionSchema(humanoidBoxCarryOptionKeys)
        ) { cfg in
            guard (cfg.options["manipulationGatedActor"] ?? 1) == 1 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-box-carry-v0 requires manipulationGatedActor=1")
            }
            return try HumanoidBoxCarryTask(configuration: .init(
                numEnvironments: cfg.numEnvironments,
                seed: cfg.seed,
                maxEpisodeSteps: Int(
                    cfg.options["maxEpisodeSteps"] ?? 600),
                solverIterations: Int(
                    cfg.options["solverIterations"] ?? 20),
                autoReset: cfg.autoReset,
                observationNoise:
                    (cfg.options["observationNoise"] ?? 1) > 0,
                minimumTrainingStationDistance:
                    cfg.options["minimumTrainingStationDistance"] ?? 0.55,
                evaluationStationDistance:
                    cfg.options["evaluationStationDistance"] ?? 1.40,
                stationDistanceCurriculumControlSteps: Int(
                    cfg.options["stationDistanceCurriculumControlSteps"]
                        ?? 12_000),
                boxMass: cfg.options["boxMass"] ?? 2,
                boxFriction: cfg.options["boxFriction"] ?? 1.2,
                minimumTrainingLiftClearance:
                    cfg.options["minimumTrainingLiftClearance"] ?? 0.001,
                liftClearance: cfg.options["liftClearance"] ?? 0.04,
                liftClearanceCurriculumControlSteps: Int(
                    cfg.options["liftClearanceCurriculumControlSteps"]
                        ?? 30_000),
                minimumTrainingCarryDistance:
                    cfg.options["minimumTrainingCarryDistance"] ?? 0.05,
                carryDistance: cfg.options["carryDistance"] ?? 0.75,
                carryDistanceCurriculumControlSteps: Int(
                    cfg.options["carryDistanceCurriculumControlSteps"]
                        ?? 30_000),
                destinationBearingCurriculumControlSteps: Int(
                    cfg.options["destinationBearingCurriculumControlSteps"]
                        ?? 0),
                minimumTrainingSuccessDwellSteps: Int(
                    cfg.options["minimumTrainingSuccessDwellSteps"] ?? 1),
                successDwellSteps: Int(
                    cfg.options["successDwellSteps"] ?? 10),
                successDwellCurriculumControlSteps: Int(
                    cfg.options["successDwellCurriculumControlSteps"]
                        ?? 30_000),
                pregraspForwardOffset:
                    cfg.options["pregraspForwardOffset"]
                        ?? HumanoidBoxCarryTask.pregraspOffset,
                pregraspLateralOffset:
                    cfg.options["pregraspLateralOffset"] ?? 0,
                approachCommandSpeed:
                    cfg.options["approachCommandSpeed"] ?? 0.40,
                carryCommandSpeed:
                    cfg.options["carryCommandSpeed"] ?? 0.30,
                carryHolonomicCommand:
                    (cfg.options["carryHolonomicCommand"] ?? 0) > 0,
                navigationGoalObservationScale:
                    cfg.options["navigationGoalObservationScale"] ?? 8,
                manipulationHandoffSteps: Int(
                    cfg.options["manipulationHandoffSteps"] ?? 24),
                carryHandoffSteps: Int(
                    cfg.options["carryHandoffSteps"] ?? 12),
                carryCommandRampSteps: Int(
                    cfg.options["carryCommandRampSteps"] ?? 12),
                manipulationArmActionScaleMultiplier:
                    cfg.options["manipulationArmActionScaleMultiplier"] ?? 2,
                carryBaseLegActionFraction:
                    cfg.options["carryBaseLegActionFraction"] ?? 0.25,
                freezeBasePolicyExpert:
                    (cfg.options["freezeBasePolicyExpert"] ?? 1) > 0,
                freezeManipulationPolicyExpert:
                    (cfg.options["freezeManipulationPolicyExpert"] ?? 1) > 0,
                freezeCarryPolicyExpert:
                    (cfg.options["freezeCarryPolicyExpert"] ?? 1) > 0,
                initializeManipulationExpertFromBaseOnTransfer:
                    (cfg.options[
                        "initializeManipulationExpertFromBaseOnTransfer"]
                        ?? 1) > 0,
                initializeCarryExpertFromManipulationExpertOnTransfer:
                    (cfg.options[
                        "initializeCarryExpertFromManipulationExpertOnTransfer"]
                        ?? 1) > 0,
                compositionalCarryController:
                    (cfg.options["compositionalCarryController"] ?? 0) > 0,
                upperBodyCarryController:
                    (cfg.options["upperBodyCarryController"] ?? 0) > 0,
                carryLocomotionControlsTorso:
                    (cfg.options["carryLocomotionControlsTorso"] ?? 0) > 0,
                initializeCarryExpertFromBaseOnTransfer:
                    (cfg.options[
                        "initializeCarryExpertFromBaseOnTransfer"] ?? 0) > 0,
                initializeCarryLocomotionExpertFromBaseOnTransfer:
                    (cfg.options[
                        "initializeCarryLocomotionExpertFromBaseOnTransfer"]
                        ?? 0) > 0,
                carryStartReplayProbability:
                    cfg.options["carryStartReplayProbability"] ?? 0,
                advanceReplaySnapshotAtDestinationContact:
                    (cfg.options[
                        "advanceReplaySnapshotAtDestinationContact"] ?? 0) > 0,
                carryArmReferenceWeight:
                    cfg.options["carryArmReferenceWeight"] ?? 1,
                carryHoldClearanceMultiplier:
                    cfg.options["carryHoldClearanceMultiplier"] ?? 1,
                carryProgressRewardWeight:
                    cfg.options["carryProgressRewardWeight"] ?? 25,
                carryLocomotionRewardMultiplier:
                    cfg.options["carryLocomotionRewardMultiplier"] ?? 1,
                carryTrackingVariance:
                    cfg.options["carryTrackingVariance"] ?? 0.25,
                coupledCarryCommandTracking:
                    (cfg.options["coupledCarryCommandTracking"] ?? 0) > 0,
                carryRootProgressRewardWeight:
                    cfg.options["carryRootProgressRewardWeight"] ?? 0,
                carryAlternatingStepRewardWeight:
                    cfg.options["carryAlternatingStepRewardWeight"] ?? 0,
                minimumLoadedAlternatingSteps: Int(
                    cfg.options["minimumLoadedAlternatingSteps"] ?? 0)))
        }
        try! registry.register(
            "humanoid-isaac-goal-v0",
            optionSchema: optionSchema(humanoidIsaacGoalOptionKeys)
        ) { cfg in
            guard (cfg.options["pointGoal"] ?? 1) == 1 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-isaac-goal-v0 requires pointGoal=1")
            }
            return try HumanoidIsaacVelocityTask(
                configuration: HumanoidIsaacVelocityTaskConfig(
                    numEnvironments: cfg.numEnvironments,
                    seed: cfg.seed,
                    maxEpisodeSteps: Int(
                        cfg.options["maxEpisodeSteps"] ?? 1_000),
                    commandResamplingSteps: Int(
                        cfg.options["commandResamplingSteps"] ?? 500),
                    standingCommandProbability: 0,
                    initialYawRange:
                        cfg.options["initialYawRange"] ?? .pi,
                    observationNoise:
                        (cfg.options["observationNoise"] ?? 1) > 0,
                    solverIterations:
                        Int(cfg.options["solverIterations"] ?? 20),
                    autoReset: cfg.autoReset,
                    pointGoal: true,
                    minimumGoalDistance:
                        cfg.options["minimumGoalDistance"] ?? 4,
                    maximumGoalDistance:
                        cfg.options["maximumGoalDistance"] ?? 8,
                    goalRadius: cfg.options["goalRadius"] ?? 0.75,
                    goalSlowdownDistance:
                        cfg.options["goalSlowdownDistance"] ?? 2.5,
                    goalCommandSpeed:
                        cfg.options["goalCommandSpeed"] ?? 0.55,
                    goalBoundaryCommandSpeed:
                        cfg.options["goalBoundaryCommandSpeed"] ?? 0.15,
                    goalDwellSteps: Int(
                        cfg.options["goalDwellSteps"] ?? 25),
                    maximumGoalArrivalSpeed:
                        cfg.options["maximumGoalArrivalSpeed"] ?? 0.25,
                    goalProgressRewardWeight:
                        cfg.options["goalProgressRewardWeight"] ?? 1,
                    goalStableRewardWeight:
                        cfg.options["goalStableRewardWeight"] ?? 2,
                    goalSuccessBonus:
                        cfg.options["goalSuccessBonus"] ?? 5,
                    projectileProbability:
                        cfg.options["projectileProbability"] ?? 0,
                    projectileCurriculumControlSteps: Int(
                        cfg.options["projectileCurriculumControlSteps"] ?? 0),
                    projectileSize:
                        cfg.options["projectileSize"] ?? 0.25,
                    projectileMass:
                        cfg.options["projectileMass"] ?? 8,
                    minimumProjectileSpeed:
                        cfg.options["minimumProjectileSpeed"] ?? 4,
                    maximumProjectileSpeed:
                        cfg.options["maximumProjectileSpeed"] ?? 6,
                    projectileLeftProbability:
                        cfg.options["projectileLeftProbability"] ?? 0.5,
                    minimumProjectileLaunchStep: Int(
                        cfg.options["minimumProjectileLaunchStep"] ?? 100),
                    maximumProjectileLaunchStep: Int(
                        cfg.options["maximumProjectileLaunchStep"] ?? 300),
                    recoveryGatedActor:
                        (cfg.options["recoveryGatedActor"] ?? 0) > 0,
                    freezeBasePolicyExpert:
                        (cfg.options["freezeBasePolicyExpert"] ?? 1) > 0,
                    recoveryContextObservations:
                        (cfg.options["recoveryContextObservations"] ?? 0) > 0,
                    recoveryContextDuration:
                        cfg.options["recoveryContextDuration"] ?? 2,
                    recoveryExpertSide:
                        cfg.options["recoveryExpertSide"] ?? 0,
                    recoveryExpertGatePeak:
                        cfg.options["recoveryExpertGatePeak"] ?? 1,
                    recoveryExpertGateDecay:
                        (cfg.options["recoveryExpertGateDecay"] ?? 0) > 0,
                    initializeRecoveryExpertFromBaseOnTransfer:
                        (cfg.options[
                            "initializeRecoveryExpertFromBaseOnTransfer"] ?? 1) > 0,
                    initializeRecoveryExpertFromMirroredBaseOnTransfer:
                        (cfg.options[
                            "initializeRecoveryExpertFromMirroredBaseOnTransfer"]
                            ?? 0) > 0,
                    postImpactUprightRewardWeight:
                        cfg.options["postImpactUprightRewardWeight"] ?? 0,
                    postImpactAngularVelocityPenaltyWeight:
                        cfg.options[
                            "postImpactAngularVelocityPenaltyWeight"] ?? 0,
                    postImpactFallPenalty:
                        cfg.options["postImpactFallPenalty"] ?? 0),
                taskID: "humanoid-isaac-goal-v0",
                includeInteractiveRobustnessProbe:
                    cfg.includeInteractiveRobustnessProbes)
        }
        try! registry.register(
            "humanoid-goal-v0",
            optionSchema: optionSchema(humanoidGoalOptionKeys)
        ) { cfg in
            guard (cfg.options["standingCommandProbability"] ?? 0) == 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-goal-v0 requires standingCommandProbability=0; "
                    + "goal arrival supplies the zero-speed command")
            }
            return try HumanoidWalkTask(
                configuration: HumanoidWalkTaskConfig(
                    numEnvironments: cfg.numEnvironments,
                    seed: cfg.seed,
                    maxEpisodeSteps: Int(
                        cfg.options["maxEpisodeSteps"] ?? 1_000),
                    controlDecimation: Int(
                        cfg.options["controlDecimation"] ?? 4),
                    minimumCommandSpeed:
                        cfg.options["minimumCommandSpeed"] ?? 0.45,
                    maximumCommandSpeed:
                        cfg.options["maximumCommandSpeed"] ?? 0.65,
                    commandCurriculumControlSteps: Int(
                        cfg.options["commandCurriculumControlSteps"] ?? 0),
                    trainingInitialEpisodeAgeFraction:
                        cfg.options["trainingInitialEpisodeAgeFraction"] ?? 0,
                    standingCommandProbability: 0,
                    commandGatedActor:
                        (cfg.options["commandGatedActor"] ?? 0) > 0,
                    threeModeActor:
                        (cfg.options["threeModeActor"] ?? 0) > 0,
                    expertGateCommandSpeed:
                        cfg.options["expertGateCommandSpeed"] ?? 0.20,
                    expertGateBlendWidth:
                        cfg.options["expertGateBlendWidth"] ?? 0,
                    standExpertBlendStartSpeed:
                        cfg.options["standExpertBlendStartSpeed"] ?? 0,
                    standExpertBlendWidth:
                        cfg.options["standExpertBlendWidth"] ?? 0,
                    standExpertRequiresDoubleSupport:
                        (cfg.options["standExpertRequiresDoubleSupport"] ?? 0) > 0,
                    standExpertUsesPlanarSpeed:
                        (cfg.options["standExpertUsesPlanarSpeed"] ?? 0) > 0,
                    freezeBasePolicyExpert:
                        (cfg.options["freezeBasePolicyExpert"] ?? 0) > 0,
                    freezeLowSpeedPolicyExpert:
                        (cfg.options["freezeLowSpeedPolicyExpert"] ?? 0) > 0,
                    trainBasePolicyExpert:
                        (cfg.options["trainBasePolicyExpert"] ?? 0) > 0,
                    velocityTrackingStandardDeviation:
                        cfg.options["velocityTrackingStandardDeviation"]
                            ?? HumanoidLocomotionObjective
                                .velocityTrackingStandardDeviation,
                    velocityTrackingErrorPenaltyWeight:
                        cfg.options["velocityTrackingErrorPenaltyWeight"] ?? 0,
                    standStillVelocityPenaltyWeight:
                        cfg.options["standStillVelocityPenaltyWeight"] ?? 2,
                    standStillJointDeviationPenaltyWeight:
                        cfg.options["standStillJointDeviationPenaltyWeight"] ?? 1,
                    standStillDoubleSupportRewardWeight:
                        cfg.options["standStillDoubleSupportRewardWeight"] ?? 1,
                    standStillFallPenalty:
                        cfg.options["standStillFallPenalty"] ?? 0,
                    lateralPenaltyWarmupControlSteps: Int(
                        cfg.options["lateralPenaltyWarmupControlSteps"] ?? 0),
                    lateralPenaltyRampControlSteps: Int(
                        cfg.options["lateralPenaltyRampControlSteps"] ?? 0),
                    laneTrackingStandardDeviation:
                        cfg.options["laneTrackingStandardDeviation"] ?? 0.30,
                    alternatingTouchdownRewardWeight:
                        cfg.options["alternatingTouchdownRewardWeight"] ?? 2,
                    flightPenaltyWeight:
                        cfg.options["flightPenaltyWeight"] ?? 1,
                    initialRollPitchRange:
                        cfg.options["initialRollPitchRange"] ?? 0.015,
                    initialYawRange:
                        cfg.options["initialYawRange"] ?? 0.05,
                    maximumGoalDirectionAngle:
                        cfg.options["maximumGoalDirectionAngle"] ?? .pi,
                    initialGoalDirectionAngle:
                        cfg.options["initialGoalDirectionAngle"] ?? 0,
                    goalDirectionCurriculumControlSteps: Int(
                        cfg.options["goalDirectionCurriculumControlSteps"]
                            ?? 2_400),
                    goalRadius: cfg.options["goalRadius"] ?? 1.5,
                    goalSlowdownDistance:
                        cfg.options["goalSlowdownDistance"] ?? 3,
                    goalObservationUsesLateralVelocity:
                        (cfg.options["goalObservationUsesLateralVelocity"] ?? 0)
                            > 0,
                    goalObservationIncludesLateralVelocity:
                        (cfg.options["goalObservationIncludesLateralVelocity"]
                            ?? 0) > 0,
                    minimumGoalDistanceMeters:
                        cfg.options["minimumGoalDistanceMeters"] ?? 0,
                    maximumGoalDistanceMeters:
                        cfg.options["maximumGoalDistanceMeters"] ?? 0,
                    initialGoalDistanceScale:
                        cfg.options["initialGoalDistanceScale"] ?? 1,
                    goalDistanceCurriculumControlSteps: Int(
                        cfg.options["goalDistanceCurriculumControlSteps"] ?? 0),
                    goalDwellSteps: Int(
                        cfg.options["goalDwellSteps"] ?? 25),
                    maximumGoalArrivalSpeed:
                        cfg.options["maximumGoalArrivalSpeed"] ?? 0.25,
                    goalBoundaryCommandSpeed:
                        cfg.options["goalBoundaryCommandSpeed"] ?? 0,
                    goalStableDwellRewardWeight:
                        cfg.options["goalStableDwellRewardWeight"] ?? 0,
                    projectileProbability:
                        cfg.options["projectileProbability"] ?? 0.5,
                    projectileCurriculumControlSteps: Int(
                        cfg.options["projectileCurriculumControlSteps"]
                            ?? 2_400),
                    minimumProjectileSpeed:
                        cfg.options["minimumProjectileSpeed"] ?? 4,
                    maximumProjectileSpeed:
                        cfg.options["maximumProjectileSpeed"] ?? 6,
                    minimumProjectileLaunchStep: Int(
                        cfg.options["minimumProjectileLaunchStep"] ?? 200),
                    maximumProjectileLaunchStep: Int(
                        cfg.options["maximumProjectileLaunchStep"] ?? 700),
                    actionTargetResponse:
                        cfg.options["actionTargetResponse"] ?? 1,
                    autoReset: cfg.autoReset),
                taskID: "humanoid-goal-v0",
                taskRevision: ((cfg.options["goalStableDwellRewardWeight"] ?? 0) > 0
                    ? (30
                        + ((cfg.options["standStillFallPenalty"] ?? 0) > 0 ? 1 : 0)
                        + ((cfg.options["freezeBasePolicyExpert"] ?? 0) > 0 ? 2 : 0)
                        + ((cfg.options["expertGateBlendWidth"] ?? 0) > 0 ? 4 : 0))
                    : (cfg.options["goalBoundaryCommandSpeed"] ?? 0) > 0
                    ? (20
                        + ((cfg.options["standStillFallPenalty"] ?? 0) > 0 ? 1 : 0)
                        + ((cfg.options["freezeBasePolicyExpert"] ?? 0) > 0 ? 2 : 0)
                        + ((cfg.options["expertGateBlendWidth"] ?? 0) > 0 ? 4 : 0))
                    : (cfg.options["expertGateBlendWidth"] ?? 0) > 0
                    ? (15
                        + ((cfg.options["standStillFallPenalty"] ?? 0) > 0 ? 1 : 0)
                        + ((cfg.options["freezeBasePolicyExpert"] ?? 0) > 0 ? 2 : 0))
                    : ((cfg.options["freezeBasePolicyExpert"] ?? 0) > 0
                        ? ((cfg.options["standStillFallPenalty"] ?? 0) > 0 ? 14 : 13)
                        : ((cfg.options["standStillFallPenalty"] ?? 0) > 0
                            ? 12
                            : ((cfg.options["commandGatedActor"] ?? 0) > 0 ? 11 : 10))))
                    + ((cfg.options["trainingInitialEpisodeAgeFraction"] ?? 0) > 0
                        ? 100 : 0)
                    + ((cfg.options["trainBasePolicyExpert"] ?? 0) > 0
                        ? 200 : 0)
                    + ((cfg.options["threeModeActor"] ?? 0) > 0
                        ? 400 : 0)
                    + ((cfg.options["velocityTrackingStandardDeviation"]
                        ?? HumanoidLocomotionObjective
                            .velocityTrackingStandardDeviation)
                        != HumanoidLocomotionObjective
                            .velocityTrackingStandardDeviation ? 800 : 0)
                    + ((cfg.options["velocityTrackingErrorPenaltyWeight"] ?? 0) > 0
                        ? 1_600 : 0)
                    + ((cfg.options["freezeLowSpeedPolicyExpert"] ?? 0) > 0
                        ? 3_200 : 0)
                    + ((cfg.options["minimumGoalDistanceMeters"] ?? 0) > 0
                        ? 6_400 : 0)
                    + ((cfg.options["standExpertBlendStartSpeed"] ?? 0) > 0
                        ? 12_800 : 0)
                    + ((cfg.options["standExpertRequiresDoubleSupport"] ?? 0) > 0
                        ? 25_600 : 0)
                    + ((cfg.options["standExpertUsesPlanarSpeed"] ?? 0) > 0
                        ? 51_200 : 0)
                    + ((cfg.options["goalObservationUsesLateralVelocity"] ?? 0)
                        > 0 ? 102_400 : 0)
                    + ((cfg.options[
                        "goalObservationIncludesLateralVelocity"] ?? 0)
                        > 0 ? 204_800 : 0))
        }
        try! registry.register(
            "arm-pusht-v0", optionSchema: optionSchema(armPushTOptionKeys)
        ) { cfg in
            return try ArmPushTTask(configuration: ArmPushTTaskConfig(
                numEnvironments: cfg.numEnvironments,
                seed: cfg.seed,
                maxEpisodeSteps: Int(cfg.options["maxEpisodeSteps"] ?? 100),
                // The maintained PushT-v1 benchmark runs policy control at
                // 20 Hz. AVBD simulates at 120 Hz, so six substeps preserve
                // both its 100-action horizon and five seconds of task time.
                controlDecimation: Int(cfg.options["controlDecimation"] ?? 6),
                autoReset: cfg.autoReset,
                jointDeltaActionScale:
                    cfg.options["jointDeltaActionScale"] ?? 0.1,
                endEffectorDeltaActionScale:
                    cfg.options["endEffectorDeltaActionScale"] ?? 0,
                linkLength1:
                    cfg.options["linkLength1"] ?? ArmPushTEnv.linkLengths.x,
                linkLength2:
                    cfg.options["linkLength2"] ?? ArmPushTEnv.linkLengths.y,
                linkMass: cfg.options["linkMass"] ?? 2.7,
                tipMass: cfg.options["tipMass"] ?? 0.75,
                motorTorque: cfg.options["motorTorque"] ?? 100,
                motorStiffness: cfg.options["motorStiffness"] ?? 1_000,
                motorDamping: cfg.options["motorDamping"] ?? 100,
                motorArmature: cfg.options["motorArmature"] ?? 0.1,
                blockMass: cfg.options["blockMass"] ?? 0.8,
                blockStaticFriction:
                    cfg.options["blockStaticFriction"] ?? 3,
                blockDynamicFriction:
                    cfg.options["blockDynamicFriction"] ?? 3,
                blockSpawnGoalBlend:
                    cfg.options["blockSpawnGoalBlend"] ?? 0,
                blockSpawnRadius: cfg.options["blockSpawnRadius"] ?? 0.07,
                blockSpawnYawRange:
                    cfg.options["blockSpawnYawRange"] ?? 0.35,
                blockSpawnLateralBias:
                    cfg.options["blockSpawnLateralBias"] ?? 0,
                goalProgressWeight: cfg.options["goalProgressWeight"] ?? 0,
                reachProgressWeight: cfg.options["reachProgressWeight"] ?? 0,
                yawProgressWeight: cfg.options["yawProgressWeight"] ?? 0,
                coverageProgressWeight:
                    cfg.options["coverageProgressWeight"] ?? 0,
                coverageRewardWeight:
                    cfg.options["coverageRewardWeight"] ?? 0,
                poseProgressRewardWeight:
                    cfg.options["poseProgressRewardWeight"] ?? 0,
                poseRewardWeight: cfg.options["poseRewardWeight"] ?? 1,
                poseRewardDistanceScale:
                    cfg.options["poseRewardDistanceScale"] ?? 5,
                reachingRewardWeight:
                    cfg.options["reachingRewardWeight"] ?? 0.05,
                actionMagnitudePenaltyWeight:
                    cfg.options["actionMagnitudePenaltyWeight"] ?? 0,
                actionRatePenaltyWeight:
                    cfg.options["actionRatePenaltyWeight"] ?? 0,
                precisionGatedActor:
                    (cfg.options["precisionGatedActor"] ?? 0) > 0,
                precisionExpertGateCoverage:
                    cfg.options["precisionExpertGateCoverage"] ?? 0.75,
                precisionExpertReleaseCoverage:
                    cfg.options["precisionExpertReleaseCoverage"] ?? 0.75,
                freezeBasePolicyExpert:
                    (cfg.options["freezeBasePolicyExpert"] ?? 0) > 0,
                pushContactProgressWeight:
                    cfg.options["pushContactProgressWeight"] ?? 0,
                reachDistancePenaltyWeight:
                    cfg.options["reachDistancePenaltyWeight"] ?? 0,
                goalDistancePenaltyWeight:
                    cfg.options["goalDistancePenaltyWeight"] ?? 0,
                yawErrorPenaltyWeight:
                    cfg.options["yawErrorPenaltyWeight"] ?? 0,
                pushContactDistancePenaltyWeight:
                    cfg.options["pushContactDistancePenaltyWeight"] ?? 0,
                successBonus: cfg.options["successBonus"] ?? 0,
                successRewardOverride:
                    cfg.options["successRewardOverride"] ?? 3,
                continueAfterSuccess:
                    (cfg.options["continueAfterSuccess"] ?? 0) > 0,
                successCoverage: cfg.options["successCoverage"]
                    ?? ArmPushTEnv.successCoverage,
                successYawTolerance: cfg.options["successYawTolerance"] ?? .pi,
                reachCurriculumSuccessDistance:
                    cfg.options["reachCurriculumSuccessDistance"] ?? 0,
                pushContactCurriculumSuccessDistance:
                    cfg.options["pushContactCurriculumSuccessDistance"] ?? 0,
                pushContactOffset: cfg.options["pushContactOffset"] ?? 0.11,
                pushContactCurriculumMaximumGoalRegression:
                    cfg.options[
                        "pushContactCurriculumMaximumGoalRegression"] ?? 0,
                pushContactCurriculumMinimumGoalProgress:
                    cfg.options[
                        "pushContactCurriculumMinimumGoalProgress"] ?? 0,
                pushContactCurriculumMaximumGoalDistance:
                    cfg.options[
                        "pushContactCurriculumMaximumGoalDistance"] ?? 0))
        }
        try! registry.register(
            "arachne15-velocity-v0",
            optionSchema: optionSchema(arachne15OptionKeys)
        ) { cfg in
            return try Arachne15LocomotionTask(
                configuration: try arachne15Configuration(
                    cfg, pointGoal: false),
                includeInteractiveRobustnessProbe:
                    cfg.includeInteractiveRobustnessProbes)
        }
        try! registry.register(
            "arachne15-goal-v0",
            optionSchema: optionSchema(arachne15OptionKeys)
        ) { cfg in
            return try Arachne15LocomotionTask(
                configuration: try arachne15Configuration(
                    cfg, pointGoal: true),
                taskID: "arachne15-goal-v0",
                includeInteractiveRobustnessProbe:
                    cfg.includeInteractiveRobustnessProbes)
        }
        try! registry.register(
            "maniskill-pusht-v1",
            optionSchema: optionSchema(maniSkillPushTOptionKeys)
        ) { cfg in
            return try ManiSkillPushTTask(configuration: .init(
                numEnvironments: cfg.numEnvironments,
                seed: cfg.seed,
                maxEpisodeSteps: Int(
                    cfg.options["maxEpisodeSteps"] ?? 100),
                controlDecimation: Int(
                    cfg.options["controlDecimation"] ?? 5),
                autoReset: cfg.autoReset,
                jointDeltaActionScale:
                    cfg.options["jointDeltaActionScale"] ?? 0.1,
                robotInitialJointNoise:
                    cfg.options["robotInitialJointNoise"] ?? 0.02,
                normalizedDenseReward:
                    (cfg.options["normalizedDenseReward"] ?? 1) > 0))
        }
        return registry
    }()
}
