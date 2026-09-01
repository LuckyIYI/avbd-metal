import MLXRL
import Foundation
import simd
import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class GEARSonicG1SessionTests: XCTestCase {
    private static let bodyNames = [
        "pelvis",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "left_wrist_roll_link", "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
        "right_wrist_roll_link", "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]

    private static let actuatorNames = [
        "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    /// Locks the physical topology extracted from NVIDIA's pinned analytic
    /// training USD. All expectations are source facts, not values loaded
    /// from the generated plant manifest beside the XML.
    func testImportedAnalyticPlantContract() throws {
        let bundle = try officialBundleDirectory(requirePolicy: false)
        let asset = try MJCFAsset.parse(
            url: bundle.appendingPathComponent("plant.xml"))

        XCTAssertEqual(asset.bodyNames, Self.bodyNames)
        XCTAssertEqual(asset.jointNames, Self.actuatorNames)
        XCTAssertEqual(asset.actuatorNames, Self.actuatorNames)
        XCTAssertEqual(asset.visualGeometryCount, 36)
        XCTAssertTrue(asset.warnings.isEmpty, "\(asset.warnings)")

        var scene = PhysicsScene(name: "gear-sonic-g1-plant-contract")
        let imported = try asset.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 1, damping: 1),
                selfCollisions: true,
                collideConnectedBodies: false,
                inertiaFrame: .principal,
                includeVisuals: false))

        XCTAssertEqual(scene.bodies.count, 30)
        XCTAssertEqual(scene.joints.count, 29)
        XCTAssertEqual(imported.actuatorJoints.count, 29)
        XCTAssertEqual(imported.actuatorNames, Self.actuatorNames)

        XCTAssertEqual(scene.colliders.count, 29)
        XCTAssertEqual(scene.colliders.filter(\.collisionEnabled).count, 29)
        XCTAssertEqual(scene.colliders.filter { $0.shape == .capsule }.count, 28)
        XCTAssertEqual(scene.colliders.filter { $0.shape == .sphere }.count, 1)
        XCTAssertEqual(scene.colliders.filter { $0.shape == .box }.count, 0)
        XCTAssertEqual(scene.rigidMeshes.count, 0)
        XCTAssertTrue(scene.colliders.allSatisfy {
            $0.convexHullVertices.isEmpty
        })

        let colliderCountsByBody = Self.bodyNames.map { name in
            let body = imported.bodiesByName[name]!
            return scene.colliders.filter { $0.body == body }.count
        }
        XCTAssertEqual(colliderCountsByBody, [
            1,
            0, 1, 0, 1, 0, 7,
            0, 1, 0, 1, 0, 7,
            0, 0, 4,
            0, 0, 1, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 1,
        ])

        let totalMass = try scene.bodies.reduce(Float.zero) { partial, body in
            partial + (try XCTUnwrap(body.mass))
        }
        XCTAssertEqual(totalMass, 33.34114202, accuracy: 1e-5)

        // With connected-body collision disabled, the exclusions must be
        // exactly the 29 articulation edges: neither missing nor broadened
        // to unrelated self-collision pairs.
        let jointPairs = Set(scene.joints.map {
            Set([$0.bodyA, $0.bodyB])
        })
        let exclusionPairs = Set(scene.collisionExclusions.map {
            Set([$0.bodyA, $0.bodyB])
        })
        XCTAssertEqual(scene.collisionExclusions.count, 29)
        XCTAssertEqual(exclusionPairs.count, 29)
        XCTAssertEqual(exclusionPairs, jointPairs)
        XCTAssertTrue(scene.joints.allSatisfy { $0.bodyA >= 0 && $0.bodyB >= 0 })

        // Loading the source-locked skin adds only render surfaces. It must
        // neither duplicate nor replace any of the 29 analytic colliders.
        var visualScene = PhysicsScene(name: "gear-sonic-g1-visual-contract")
        _ = try asset.instantiate(
            in: &visualScene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 1, damping: 1),
                selfCollisions: true,
                collideConnectedBodies: false,
                inertiaFrame: .principal,
                includeVisuals: true))
        XCTAssertEqual(visualScene.bodies.count, 30)
        XCTAssertEqual(visualScene.joints.count, 29)
        XCTAssertEqual(visualScene.colliders.count, 29)
        XCTAssertEqual(
            visualScene.colliders.filter(\.collisionEnabled).count, 29)
        XCTAssertTrue(visualScene.colliders.allSatisfy { !$0.isRendered })
        XCTAssertEqual(visualScene.rigidMeshes.count, 36)
    }

    /// Exercises the physical reset path without MLX. The released dance
    /// clip's frame-zero pose is in policy order, so this also locks the
    /// policy-to-actuator permutation and proves source coordinates survive
    /// the home-rebased solver boundary exactly once.
    func testOfficialFrameZeroResetDoesNotDoubleApplyHomeOffset() throws {
        let bundle = try officialBundleDirectory(requirePolicy: true)
        let referenceDirectory = try officialReferenceDirectory()
        let manifest = try policyManifest(in: bundle)
        let reference = try GEARSonicG1ReferenceClip(
            directory: referenceDirectory.path)
        let environment = try makeEnvironment(
            bundle: bundle, control: manifest.control)
        let expected = frameZeroActuatorPositions(
            reference: reference, control: manifest.control)

        let root = reference.bodyPositions[0]
        let quaternion = try reference.rootQuaternionWXYZ(at: 0)
        try environment.reset([.init(
            sourceJointPositions: expected,
            rootPosition: F3(Float(root[0]), Float(root[1]), Float(root[2])),
            rootRotation: Quat(
                real: Float(quaternion.w),
                imag: F3(
                    Float(quaternion.x), Float(quaternion.y),
                    Float(quaternion.z))).normalized)])

        let actual = try XCTUnwrap(environment.states().first).jointAngles
        XCTAssertEqual(actual.count, 29)
        XCTAssertGreaterThan(zip(expected, manifest.control.defaultJointPositions)
            .filter { abs($0.0 - $0.1) > 0.05 }.count, 10)
        for actuator in 0..<29 {
            XCTAssertEqual(
                actual[actuator], expected[actuator], accuracy: 2e-5,
                "source joint \(Self.actuatorNames[actuator]) was home-offset twice")
        }
    }

    /// Covers the same frame-zero invariant through the public session
    /// initializer, including imported MLX golden verification. SwiftPM does
    /// not package mlx-swift's default.metallib, so it intentionally skips
    /// there and runs under the Xcode package test bundle.
    func testSessionInitializesAtOfficialFrameZero() throws {
        try requirePackagedMLXMetalLibrary()
        let bundle = try officialBundleDirectory(requirePolicy: true)
        let referenceDirectory = try officialReferenceDirectory()
        let reference = try GEARSonicG1ReferenceClip(
            directory: referenceDirectory.path)
        let manifest = try policyManifest(in: bundle)
        let expected = frameZeroActuatorPositions(
            reference: reference, control: manifest.control)

        let session = try GEARSonicG1Session(
            bundleDirectory: bundle.path,
            referenceDirectory: referenceDirectory.path,
            environmentCount: 1,
            initializeFromReference: true)
        XCTAssertEqual(session.referenceFrame, 0)
        XCTAssertEqual(session.controlSteps, 0)
        XCTAssertTrue(session.isPlaying)
        XCTAssertFalse(session.completedReference)
        XCTAssertTrue(session.policyVerification.passed)

        let initialState = try XCTUnwrap(session.environment.states().first)
        let actual = initialState.jointAngles
        for actuator in 0..<29 {
            XCTAssertEqual(
                actual[actuator], expected[actuator], accuracy: 2e-5,
                "session joint \(Self.actuatorNames[actuator]) did not match frame zero")
        }

        let stepped = try XCTUnwrap(session.step().first)
        XCTAssertEqual(session.referenceFrame, 1)
        XCTAssertEqual(session.controlSteps, 1)
        XCTAssertTrue([
            stepped.root.position.x, stepped.root.position.y,
            stepped.root.position.z,
        ].allSatisfy(\.isFinite))
        XCTAssertTrue(stepped.jointAngles.allSatisfy(\.isFinite))
        XCTAssertTrue(stepped.jointVelocities.allSatisfy(\.isFinite))

        // One transition commands frame zero, then advances the source
        // evaluation clock. The trajectory endpoint must therefore be the
        // heading-aligned frame-one displacement, not frame zero again.
        let initialRotation = initialState.root.rotation
        let alignment = try reference.headingAlignment(
            robotInitialBaseQuaternionWXYZ: .init(
                w: Double(initialRotation.real),
                x: Double(initialRotation.imag.x),
                y: Double(initialRotation.imag.y),
                z: Double(initialRotation.imag.z)),
            referenceFrame: 0)
        let alignmentWXYZ = alignment.referenceAlignmentQuaternionWXYZ
        let alignmentRotation = simd_quatf(
            ix: Float(alignmentWXYZ.x),
            iy: Float(alignmentWXYZ.y),
            iz: Float(alignmentWXYZ.z),
            r: Float(alignmentWXYZ.w))
        let frameZeroRoot = reference.bodyPositions[0]
        let frameOneRoot = reference.bodyPositions[1]
        let expectedReferenceDisplacement = alignmentRotation.act(F3(
            Float(frameOneRoot[0] - frameZeroRoot[0]),
            Float(frameOneRoot[1] - frameZeroRoot[1]),
            Float(frameOneRoot[2] - frameZeroRoot[2])))

        let report = session.report()
        XCTAssertEqual(report.schemaVersion, 3)
        XCTAssertEqual(report.environmentCount, 1)
        XCTAssertEqual(report.trajectoryEnvironmentIndex, 0)
        XCTAssertTrue(report.sourceCriteriaPassed)
        XCTAssertFalse(report.jointVelocityLimitsEnforced)
        XCTAssertEqual(report.referenceRootDisplacementMeters.count, 3)
        XCTAssertEqual(
            report.referenceRootDisplacementMeters[0],
            expectedReferenceDisplacement.x, accuracy: 1e-6)
        XCTAssertEqual(
            report.referenceRootDisplacementMeters[1],
            expectedReferenceDisplacement.y, accuracy: 1e-6)
        XCTAssertEqual(
            report.referenceRootDisplacementMeters[2],
            expectedReferenceDisplacement.z, accuracy: 1e-6)
        XCTAssertTrue(report.policyParityPassed)
    }

    func testReusableRobustnessBoxesHavePhysicalReplicaOwnership() throws {
        let bundle = try officialBundleDirectory(requirePolicy: true)
        let manifest = try policyManifest(in: bundle)
        let environment = try makeEnvironment(
            bundle: bundle, control: manifest.control,
            environmentCount: 2, projectileSize: 0.30,
            projectileMass: 12)

        XCTAssertEqual(environment.refs.count, 2)
        for replica in environment.refs.indices {
            let reference = environment.refs[replica]
            XCTAssertFalse(reference.bodies.contains(reference.projectile))
            XCTAssertEqual(
                environment.solver.bodyMass(reference.projectile), 12,
                accuracy: 1e-5)
            let colliders = environment.scene.colliders.filter {
                $0.body == reference.projectile
            }
            XCTAssertEqual(colliders.count, 1)
            XCTAssertEqual(colliders[0].shape, .box)
            XCTAssertEqual(colliders[0].size, F3(repeating: 0.30))
            XCTAssertEqual(colliders[0].collisionGroup, UInt32(replica + 1))
            XCTAssertTrue(colliders[0].collisionEnabled)
            let robotColliders = environment.scene.colliders.filter {
                reference.bodies.contains($0.body)
            }
            XCTAssertEqual(robotColliders.count, 29)
            XCTAssertTrue(robotColliders.allSatisfy {
                $0.collisionEnabled
                    && $0.collisionGroup == UInt32(replica + 1)
            }, "box and every G1 primitive must share replica ownership")
            let hidden = environment.solver.bodyStates([
                reference.projectile,
            ])[0]
            XCTAssertEqual(hidden.position.z, -4, accuracy: 1e-6)
            XCTAssertEqual(simd_length(hidden.linearVelocity), 0, accuracy: 1e-6)
            XCTAssertEqual(simd_length(hidden.angularVelocity), 0, accuracy: 1e-6)
        }

        environment.throwBoxes(
            environmentIDs: [0, 1], sideSigns: [1, -1],
            launchDistance: 1.2, speed: 6)
        let launched = environment.solver.bodyStates(
            environment.refs.map(\.projectile))
        let humanoids = environment.states()
        for replica in launched.indices {
            XCTAssertGreaterThan(launched[replica].position.z, 0.4)
            XCTAssertGreaterThan(simd_length(launched[replica].linearVelocity), 5)
            XCTAssertGreaterThan(
                simd_dot(
                    launched[replica].linearVelocity,
                    humanoids[replica].torso.position
                        - launched[replica].position),
                0, "box must travel toward its measured torso")
        }

        environment.hideBoxes(environmentIDs: [0])
        var hidden = environment.solver.bodyStates(
            environment.refs.map(\.projectile))
        XCTAssertEqual(hidden[0].position.z, -4, accuracy: 1e-6)
        XCTAssertEqual(simd_length(hidden[0].linearVelocity), 0, accuracy: 1e-6)
        XCTAssertGreaterThan(hidden[1].position.z, 0.4)

        let resets = environment.environmentOrigins.map { origin in
            GEARSonicG1Sim2SimEnv.ResetState(
                sourceJointPositions: manifest.control.defaultJointPositions,
                rootPosition: origin,
                rootRotation: Quat(real: 1, imag: .zero))
        }
        try environment.reset(resets)
        hidden = environment.solver.bodyStates(
            environment.refs.map(\.projectile))
        for state in hidden {
            XCTAssertEqual(state.position.z, -4, accuracy: 1e-6)
            XCTAssertEqual(simd_length(state.linearVelocity), 0, accuracy: 1e-6)
            XCTAssertEqual(simd_length(state.angularVelocity), 0, accuracy: 1e-6)
        }
    }

    func testRobustnessBoxTransfersMomentumThroughContactSolver() throws {
        let bundle = try officialBundleDirectory(requirePolicy: true)
        let manifest = try policyManifest(in: bundle)
        let environment = try makeEnvironment(
            bundle: bundle, control: manifest.control,
            environmentCount: 2)
        let resets = environment.environmentOrigins.map { origin in
            GEARSonicG1Sim2SimEnv.ResetState(
                sourceJointPositions: manifest.control.defaultJointPositions,
                rootPosition: origin,
                rootRotation: Quat(real: 1, imag: .zero))
        }
        try environment.reset(resets)
        environment.throwBoxes(
            environmentIDs: [0], sideSigns: [1],
            launchDistance: 1.2, speed: 6)

        let projectile = environment.refs[0].projectile
        let initialVelocity = environment.solver.bodyVelocity(projectile)
        let incomingDirection = simd_normalize(initialVelocity)
        let targets = ContiguousArray(
            (0..<2).flatMap { _ in
                manifest.control.defaultJointPositions
            })
        var contacted = false
        var projectileSpeedAtContact = Float.greatestFiniteMagnitude
        var torsoVelocityDeltaAtContact: Float = 0
        for _ in 0..<80 where !contacted {
            try environment.step(
                sourceJointPositionTargets: targets, decimation: 1)
            contacted = environment.boxRobotContacts()[0]
            if contacted {
                projectileSpeedAtContact = simd_dot(
                    environment.solver.bodyVelocity(projectile),
                    incomingDirection)
                let states = environment.states()
                torsoVelocityDeltaAtContact = simd_dot(
                    states[0].torso.linearVelocity
                        - states[1].torso.linearVelocity,
                    incomingDirection)
            }
        }

        XCTAssertTrue(contacted,
                      "thrown box must form a physical G1 contact manifold")
        XCTAssertLessThan(
            projectileSpeedAtContact,
            simd_dot(initialVelocity, incomingDirection) - 0.1,
            "box must lose incoming momentum through collision")
        XCTAssertGreaterThan(
            torsoVelocityDeltaAtContact, 0.01,
            "impacted replica must receive momentum absent from its control replica")
        XCTAssertFalse(environment.boxRobotContacts()[1],
                       "collision groups must isolate the untouched replica")
    }

    private func makeEnvironment(
        bundle: URL, control: GEARSonicG1PolicyManifest.Control,
        environmentCount: Int = 1,
        projectileSize: Float = 0.25,
        projectileMass: Float = 8
    ) throws -> GEARSonicG1Sim2SimEnv {
        try GEARSonicG1Sim2SimEnv(
            plantURL: bundle.appendingPathComponent("plant.xml"),
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
                solverIterations: 8,
                projectileSize: projectileSize,
                projectileMass: projectileMass,
                includeVisuals: false))
    }

    private func frameZeroActuatorPositions(
        reference: GEARSonicG1ReferenceClip,
        control: GEARSonicG1PolicyManifest.Control
    ) -> [Float] {
        let policyPositions = reference.jointPositionsPolicyOrder[0]
        return (0..<29).map { actuator in
            Float(policyPositions[control.actuatorToPolicy[actuator]])
        }
    }

    private func policyManifest(
        in directory: URL
    ) throws -> GEARSonicG1PolicyManifest {
        try JSONDecoder().decode(
            GEARSonicG1PolicyManifest.self,
            from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
    }

    private func officialBundleDirectory(requirePolicy: Bool) throws -> URL {
        let root = TestPaths.repositoryRoot
        let directory = root.appendingPathComponent(
            "checkpoints/external/gear-sonic-g1", isDirectory: true)
        var required = ["plant.xml"]
        if requirePolicy {
            required += [
                "manifest.json", "policy.safetensors",
                "LICENSE.nvidia-model.txt",
            ]
        }
        guard required.allSatisfy({ name in
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(name).path)
        }) else {
            throw XCTSkip(
                "official GEAR-SONIC bundle is absent; run both import tools first")
        }
        return directory
    }

    private func officialReferenceDirectory() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        let root = TestPaths.repositoryRoot
        let candidates = [
            environment["AVBD_GEAR_SONIC_REFERENCE_DIR"],
            root.appendingPathComponent(
                "checkpoints/external/gear-sonic-g1/references/"
                    + "dance_in_da_party_001__A464",
                isDirectory: true).path,
        ].compactMap { $0 }.map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        let required = [
            "joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv",
        ]
        guard let directory = candidates.first(where: { candidate in
            required.allSatisfy { name in
                FileManager.default.fileExists(
                    atPath: candidate.appendingPathComponent(name).path)
            }
        }) else {
            throw XCTSkip(
                "official GEAR-SONIC reference is absent; set "
                    + "AVBD_GEAR_SONIC_REFERENCE_DIR")
        }
        return directory
    }

    private func requirePackagedMLXMetalLibrary() throws {
        let library = Bundle(for: Self.self).bundleURL
            .appendingPathComponent("Contents/Resources/mlx-swift_Cmlx.bundle")
            .appendingPathComponent("Contents/Resources/default.metallib")
        guard FileManager.default.fileExists(atPath: library.path) else {
            throw XCTSkip("requires an Xcode-packaged MLX default.metallib")
        }
    }
}
