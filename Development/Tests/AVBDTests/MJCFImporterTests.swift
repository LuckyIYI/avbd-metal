import Foundation
import simd
import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class MJCFImporterTests: XCTestCase {
    func testInstantiationOptionsDefaults() {
        let options = MJCFInstantiationOptions()
        XCTAssertEqual(options.worldOffset, .zero)
        XCTAssertEqual(
            options.defaultMotorGain,
            MJCFMotorGain(stiffness: 0, damping: 0))
        XCTAssertTrue(options.motorGains.isEmpty)
        XCTAssertTrue(options.motorEffortLimits.isEmpty)
        XCTAssertTrue(options.jointArmatures.isEmpty)
        XCTAssertTrue(options.jointHomePositions.isEmpty)
        XCTAssertFalse(options.fixedBase)
        XCTAssertEqual(options.gravityScale, 1)
        XCTAssertEqual(options.collisionGroup, 0)
        XCTAssertTrue(options.selfCollisions)
        XCTAssertTrue(options.collideConnectedBodies)
        XCTAssertEqual(options.inertiaFrame, .principal)
        XCTAssertEqual(options.dynamicsScale, .identity)
        XCTAssertTrue(options.includeVisuals)
    }

    func testInstantiationOptionsApplyCombinedConfiguration() throws {
        let xml = """
        <mujoco model="options-fixture">
          <worldbody>
            <body name="root" pos="0 0 1">
              <inertial pos="0.2 0 0"
                        quat="0.70710678 0 0 0.70710678"
                        mass="2" diaginertia="1 2 3"/>
              <geom type="box" size="0.5 0.5 0.5" friction="0.4"/>
              <geom type="sphere" size="0.2"
                    contype="0" conaffinity="0"/>
              <body name="child" pos="0 0 1">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="hinge" axis="0 1 0" range="-1 1"
                       damping="2" armature="0.1"/>
                <geom type="sphere" size="0.1" friction="0.5"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="drive" joint="hinge" ctrlrange="-4 4"/>
          </actuator>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))
        let options = MJCFInstantiationOptions(
            worldOffset: F3(1, 2, 3),
            defaultMotorGain: .init(stiffness: 10, damping: 3),
            motorGains: ["drive": .init(stiffness: 20, damping: 4)],
            jointHomePositions: ["hinge": 0.25],
            fixedBase: true,
            gravityScale: 0.25,
            collisionGroup: 12,
            selfCollisions: false,
            inertiaFrame: .linkAligned,
            dynamicsScale: .init(
                mass: 1.5, inertia: 0.8, friction: 1.25,
                motorTorque: 0.9, motorStiffness: 1.1,
                motorDamping: 1.2, armature: 1.3),
            includeVisuals: false)

        var scene = PhysicsScene(name: "combined-options")
        let imported = try asset.instantiate(in: &scene, options: options)
        let root = scene.bodies[imported.rootBody]
        XCTAssertEqual(root.position, F3(1.2, 2, 4))
        XCTAssertEqual(root.rotation.real, 1, accuracy: 1e-6)
        XCTAssertLessThan(simd_length(root.rotation.imag), 1e-6)
        XCTAssertEqual(try XCTUnwrap(root.mass), 3, accuracy: 1e-6)
        XCTAssertLessThan(
            simd_length(try XCTUnwrap(root.diagonalInertia)
                - F3(1.2, 2.4, 3.6)),
            1e-6)
        XCTAssertTrue(scene.bodies.allSatisfy { $0.gravityScale == 0.25 })
        XCTAssertEqual(scene.colliders.count, 2)
        XCTAssertTrue(scene.colliders.allSatisfy {
            $0.collisionGroup == 12 && $0.isRendered
        })
        XCTAssertEqual(scene.colliders[0].friction, 0.5, accuracy: 1e-6)
        XCTAssertEqual(scene.collisionExclusions.count, 1)

        XCTAssertEqual(scene.joints.count, 2)
        XCTAssertEqual(scene.joints[0].bodyA, -1)
        let hingeIndex = try XCTUnwrap(imported.jointsByName["hinge"])
        let hinge = scene.joints[hingeIndex]
        XCTAssertEqual(hinge.motorTorque, 3.6, accuracy: 1e-6)
        XCTAssertEqual(hinge.motorStiffness, 22, accuracy: 1e-6)
        XCTAssertEqual(hinge.motorDamping, 4.8, accuracy: 1e-6)
        XCTAssertEqual(hinge.armature, 0.13, accuracy: 1e-6)
        XCTAssertEqual(hinge.limitLo, -1.25, accuracy: 1e-6)
        XCTAssertEqual(hinge.limitHi, 0.75, accuracy: 1e-6)
    }

    func testBundledArachneProfilesSeparateVisualsFromCollision() throws {
        let training = try MJCFAsset.bundledArachne15(profile: .training)
        let validation = try MJCFAsset.bundledArachne15(profile: .validation)

        XCTAssertEqual(training.bodyNames.count, 17)
        XCTAssertEqual(training.jointNames.count, 16)
        XCTAssertEqual(training.actuatorNames.count, 16)
        XCTAssertEqual(training.actuatorNames, validation.actuatorNames)
        XCTAssertEqual(training.bodyNames, validation.bodyNames)
        XCTAssertEqual(training.visualGeometryCount, 48)
        XCTAssertEqual(validation.visualGeometryCount, 48)
        XCTAssertTrue(training.warnings.isEmpty, "\(training.warnings)")
        XCTAssertTrue(validation.warnings.isEmpty, "\(validation.warnings)")

        var trainingScene = PhysicsScene(name: "arachne-training")
        let imported = try training.instantiate(
            in: &trainingScene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 2, damping: 0.08),
                collisionGroup: 7,
                selfCollisions: false))
        XCTAssertEqual(trainingScene.bodies.count, 17)
        XCTAssertEqual(trainingScene.joints.count, 16)
        XCTAssertEqual(trainingScene.rigidMeshes.count, 28)
        XCTAssertEqual(trainingScene.colliders.filter(\.collisionEnabled).count, 39)
        XCTAssertEqual(trainingScene.colliders.filter { !$0.collisionEnabled }.count, 20)
        XCTAssertTrue(trainingScene.colliders.filter(\.collisionEnabled)
            .allSatisfy { $0.collisionGroup == 7 && !$0.isRendered })
        XCTAssertEqual(imported.actuatorJoints.count, 16)
        for name in training.bodyNames where name.hasSuffix("_tibia") {
            let body = try XCTUnwrap(imported.bodiesByName[name])
            let foot = try XCTUnwrap(trainingScene.colliders.first {
                $0.body == body && $0.collisionEnabled && $0.shape == .box
                    && abs($0.friction - 0.9) < 1e-6
            })
            XCTAssertEqual(foot.size.x, 0.0175, accuracy: 1e-7)
            XCTAssertEqual(foot.size.y, 0.0140, accuracy: 1e-7)
            XCTAssertEqual(foot.size.z, 0.0080, accuracy: 1e-7)
        }

        var validationScene = PhysicsScene(name: "arachne-validation")
        _ = try validation.instantiate(
            in: &validationScene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 2, damping: 0.08)))
        XCTAssertEqual(validationScene.colliders.filter(\.collisionEnabled).count, 60)
        XCTAssertEqual(validationScene.rigidMeshes.count, 28)

        let solver = try GPUSolver(scene: trainingScene)
        let surface = try XCTUnwrap(solver.renderRigidMeshSurface)
        XCTAssertGreaterThan(surface.vertexCount, 0)
    }

    func testBundledUnitreeRLGymTransferPlantContract() throws {
        let asset = try MJCFAsset.bundledUnitreeRLGymH1()
        XCTAssertEqual(asset.name, "unitree_rl_gym_h1")
        XCTAssertEqual(asset.bodyNames.count, 11)
        XCTAssertEqual(asset.jointNames.count, 10)
        XCTAssertEqual(asset.actuatorNames, [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle", "right_hip_yaw",
            "right_hip_roll", "right_hip_pitch", "right_knee",
            "right_ankle",
        ])

        var scene = PhysicsScene(name: "unitree-rl-gym-transfer")
        let imported = try asset.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 100, damping: 2),
                selfCollisions: false,
                inertiaFrame: .principal))
        XCTAssertEqual(scene.bodies.count, 11)
        XCTAssertEqual(scene.joints.count, 10)
        XCTAssertEqual(imported.actuatorJoints.count, 10)
        XCTAssertEqual(scene.collisionExclusions.count, 55)

        let pelvis = try XCTUnwrap(imported.bodiesByName["pelvis"])
        XCTAssertEqual(try XCTUnwrap(scene.bodies[pelvis].mass),
                       29.847, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(scene.bodies[pelvis].diagonalInertia),
                       F3(1.32807, 0.960733, 0.53795))

        let leftFoot = try XCTUnwrap(
            imported.bodiesByName["left_ankle_link"])
        let foot = try XCTUnwrap(
            scene.colliders.first(where: { $0.body == leftFoot }))
        XCTAssertEqual(foot.shape, .box)
        XCTAssertEqual(foot.size, F3(0.30, 0.08, 0.02))
        XCTAssertEqual(scene.joints[
            try XCTUnwrap(imported.jointsByName["left_ankle"])
        ].motorTorque, 40, accuracy: 1e-6)

        XCTAssertTrue(imported.actuatorJoints.allSatisfy {
            scene.joints[$0].motorMode == .implicitPositionPD
        })

        let transfer = try UnitreeH1Sim2SimEnv()
        XCTAssertEqual(transfer.scene.settings.dt, 0.002)
        XCTAssertEqual(transfer.refs.motors.count, 10)
        XCTAssertTrue(transfer.refs.motors.allSatisfy {
            transfer.scene.joints[$0].motorMode == .explicitTorquePD
        })
    }

    /// Integration check against the vendored MuJoCo Menagerie H1 dynamics
    /// asset, including the exact source topology and foot collision shapes.
    func testBundledOfficialMenagerieH1() throws {
        let asset = try MJCFAsset.bundledUnitreeH1()
        XCTAssertEqual(asset.name, "h1")
        XCTAssertEqual(asset.bodyNames.count, 20)
        XCTAssertEqual(asset.jointNames.count, 19)
        XCTAssertEqual(asset.actuatorNames.count, 19)

        var scene = PhysicsScene(name: "official-h1-import")
        let imported = try asset.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 100, damping: 5)))
        XCTAssertEqual(scene.bodies.count, 20)
        XCTAssertEqual(scene.joints.count, 19)
        XCTAssertEqual(scene.colliders.count, 32)
        XCTAssertEqual(scene.collisionExclusions.count, 2)
        XCTAssertEqual(imported.actuatorJoints.count, 19)

        let pelvis = try XCTUnwrap(imported.bodiesByName["pelvis"])
        XCTAssertEqual(try XCTUnwrap(scene.bodies[pelvis].mass),
                       5.39, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(scene.bodies[pelvis].diagonalInertia).x,
                       0.0490211, accuracy: 1e-7)
        XCTAssertEqual(scene.bodies[pelvis].position.z, 1.01478, accuracy: 1e-6)

        let leftFoot = try XCTUnwrap(imported.bodiesByName["left_ankle_link"])
        let rightFoot = try XCTUnwrap(imported.bodiesByName["right_ankle_link"])
        XCTAssertEqual(scene.colliders.filter { $0.body == leftFoot }.count, 3)
        XCTAssertEqual(scene.colliders.filter { $0.body == rightFoot }.count, 3)
        XCTAssertEqual(scene.joints[try XCTUnwrap(imported.jointsByName["left_knee"])]
            .motorTorque, 300, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[try XCTUnwrap(imported.jointsByName["left_knee"])]
            .armature, 0.1, accuracy: 1e-6)

        // This also compiles the collider-aware Metal kernels for the real
        // 20-link/32-primitive topology.
        XCTAssertNoThrow(try GPUSolver(scene: scene))
    }

    func testDefaultsFromToAndInertialFrameTransform() throws {
        let xml = """
        <mujoco model="fixture">
          <default>
            <default class="robot">
              <joint damping="2"/>
              <default class="collision"><geom type="capsule" size="0.1"/></default>
            </default>
          </default>
          <worldbody>
            <body name="root" pos="0 0 1" childclass="robot">
              <inertial pos="0.2 0 0" quat="1 0 0 0" mass="2" diaginertia="1 2 3"/>
              <freejoint/>
              <geom class="collision" fromto="0 0 0 0 0 1"/>
              <body name="child" pos="0 0 1">
                <inertial pos="0 0 0" quat="1 0 0 0" mass="1" diaginertia="1 1 1"/>
                <joint name="hinge" axis="0 1 0" range="-1 1"/>
                <geom class="collision" fromto="0 0 0 1 0 0"/>
              </body>
            </body>
          </worldbody>
          <actuator><motor name="hinge" joint="hinge" ctrlrange="-7 7"/></actuator>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))
        var missingGainScene = PhysicsScene(name: "missing-gain")
        XCTAssertThrowsError(try asset.instantiate(
            in: &missingGainScene,
            options: MJCFInstantiationOptions())) {
            XCTAssertTrue(String(describing: $0).contains(
                "positive position-PD stiffness"))
        }
        var scene = PhysicsScene(name: "fixture")
        let imported = try asset.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 20, damping: 3)))

        XCTAssertEqual(scene.bodies.count, 2)
        XCTAssertEqual(scene.colliders.count, 2)
        XCTAssertEqual(scene.joints.count, 1)
        XCTAssertEqual(scene.bodies[imported.rootBody].position, F3(0.2, 0, 1))
        XCTAssertEqual(scene.colliders[0].localPosition, F3(-0.2, 0, 0.5))
        XCTAssertEqual(scene.colliders[0].size.x, 1, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].motorTorque, 7, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].motorDamping, 3, accuracy: 1e-6)
    }

    func testReplicaDynamicsScaleIsAppliedAtImportBoundary() throws {
        let xml = """
        <mujoco model="scale-fixture">
          <worldbody>
            <body name="root">
              <inertial mass="2" diaginertia="1 2 3"/>
              <geom type="box" size="0.5 0.5 0.5" friction="0.4"/>
              <body name="child" pos="0 0 1">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="hinge" damping="2" armature="0.1"/>
                <geom type="sphere" size="0.1" friction="0.5"/>
              </body>
            </body>
          </worldbody>
          <actuator><motor name="hinge" joint="hinge" ctrlrange="-4 4"/></actuator>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))
        var scene = PhysicsScene(name: "scaled")
        let imported = try asset.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 10, damping: 3),
                collisionGroup: 11,
                dynamicsScale: .init(
                    mass: 1.5, inertia: 0.8, friction: 1.25,
                    motorTorque: 0.9, motorStiffness: 1.1,
                    motorDamping: 1.2, armature: 1.3)))
        let root = scene.bodies[imported.rootBody]
        XCTAssertEqual(try XCTUnwrap(root.mass), 3, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(root.diagonalInertia).x, 1.2,
                       accuracy: 1e-6)
        XCTAssertEqual(scene.colliders[0].friction, 0.5, accuracy: 1e-6)
        XCTAssertTrue(scene.colliders.allSatisfy { $0.collisionGroup == 11 })
        XCTAssertEqual(scene.joints[0].motorTorque, 3.6, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].motorStiffness, 11, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].motorDamping, 3.6, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].armature, 0.13, accuracy: 1e-6)
    }

    func testTopLevelFreeJointUsesAlreadyFreeRootAndNestedFreeIsRejected() throws {
        let xml = """
        <mujoco model="free-root">
          <worldbody>
            <body name="root">
              <inertial mass="2" diaginertia="1 1 1"/>
              <joint name="root_free" type="free"/>
              <body name="child" pos="0 0 1">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="hinge" type="hinge"/>
              </body>
            </body>
          </worldbody>
          <actuator><motor name="drive" joint="hinge"/></actuator>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))
        XCTAssertEqual(asset.jointNames, ["hinge"])
        var scene = PhysicsScene(name: "free-root")
        let imported = try asset.instantiate(
            in: &scene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 1, damping: 0)))
        XCTAssertEqual(scene.bodies.count, 2)
        XCTAssertEqual(scene.joints.count, 1)
        XCTAssertEqual(imported.actuatorJoints, [0])

        let nestedFree = """
        <mujoco model="nested-free">
          <worldbody>
            <body name="root">
              <inertial mass="1" diaginertia="1 1 1"/>
              <body name="child">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint type="free"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        XCTAssertThrowsError(try MJCFAsset.parse(data: Data(nestedFree.utf8))) {
            XCTAssertTrue(String(describing: $0).contains(
                "free joint on child must be on a top-level body"))
        }
    }

    func testMotorEffortPrecedenceAndNamedPhysicalOverrides() throws {
        let xml = """
        <mujoco model="effort-precedence">
          <worldbody>
            <body name="root">
              <inertial mass="1" diaginertia="1 1 1"/>
              <body name="force_body">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="force_joint" armature="0.1"
                       actuatorfrcrange="-9 9"/>
              </body>
              <body name="joint_body">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="joint_limit" armature="0.2"
                       actuatorfrcrange="-6 8"/>
              </body>
              <body name="control_body">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="control_joint" armature="0.3"/>
              </body>
              <body name="legacy_body">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="legacy_joint" armature="0.4"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="force_motor" joint="force_joint"
                   forcerange="-7 7" ctrlrange="-3 3"/>
            <motor name="joint_motor" joint="joint_limit" ctrlrange="-3 3"/>
            <motor name="control_motor" joint="control_joint" ctrlrange="-5 4"/>
            <motor name="legacy_motor" joint="legacy_joint"/>
          </actuator>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))
        var sourceScene = PhysicsScene(name: "source-effort")
        let source = try asset.instantiate(
            in: &sourceScene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 1, damping: 0)))
        XCTAssertEqual(sourceScene.joints[
            try XCTUnwrap(source.jointsByName["force_joint"])
        ].motorTorque, 7, accuracy: 1e-6)
        XCTAssertEqual(sourceScene.joints[
            try XCTUnwrap(source.jointsByName["joint_limit"])
        ].motorTorque, 8, accuracy: 1e-6)
        XCTAssertEqual(sourceScene.joints[
            try XCTUnwrap(source.jointsByName["control_joint"])
        ].motorTorque, 5, accuracy: 1e-6)
        XCTAssertEqual(sourceScene.joints[
            try XCTUnwrap(source.jointsByName["legacy_joint"])
        ].motorTorque, 1, accuracy: 1e-6)

        var overrideScene = PhysicsScene(name: "override-effort")
        let overridden = try asset.instantiate(
            in: &overrideScene,
            options: MJCFInstantiationOptions(
                defaultMotorGain: .init(stiffness: 1, damping: 0),
                motorEffortLimits: [
                    "force_motor": 11,
                    "force_joint": 99,
                    "joint_limit": 12,
                ],
                jointArmatures: [
                    "force_motor": 1.1,
                    "force_joint": 9.9,
                    "joint_limit": 1.2,
                ]))
        let forceJoint = overrideScene.joints[
            try XCTUnwrap(overridden.jointsByName["force_joint"])]
        XCTAssertEqual(forceJoint.motorTorque, 11, accuracy: 1e-6)
        XCTAssertEqual(forceJoint.armature, 1.1, accuracy: 1e-6)
        let jointLimit = overrideScene.joints[
            try XCTUnwrap(overridden.jointsByName["joint_limit"])]
        XCTAssertEqual(jointLimit.motorTorque, 12, accuracy: 1e-6)
        XCTAssertEqual(jointLimit.armature, 1.2, accuracy: 1e-6)
    }

    func testConnectedBodyCollisionFilteringIsExplicit() throws {
        let xml = """
        <mujoco model="connected-filter">
          <worldbody>
            <body name="root">
              <inertial mass="1" diaginertia="1 1 1"/>
              <body name="hinged">
                <inertial mass="1" diaginertia="1 1 1"/>
                <joint name="hinge"/>
                <body name="nested_without_joint">
                  <inertial mass="1" diaginertia="1 1 1"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))

        var legacyScene = PhysicsScene(name: "connected-legacy")
        _ = try asset.instantiate(
            in: &legacyScene, options: MJCFInstantiationOptions())
        XCTAssertTrue(legacyScene.collisionExclusions.isEmpty)

        var filteredScene = PhysicsScene(name: "connected-filtered")
        let imported = try asset.instantiate(
            in: &filteredScene,
            options: MJCFInstantiationOptions(
                selfCollisions: true, collideConnectedBodies: false))
        XCTAssertEqual(filteredScene.collisionExclusions.count, 2)
        let pairs = Set(filteredScene.collisionExclusions.map {
            Set([$0.bodyA, $0.bodyB])
        })
        XCTAssertEqual(pairs, Set([
            Set([
                try XCTUnwrap(imported.bodiesByName["root"]),
                try XCTUnwrap(imported.bodiesByName["hinged"]),
            ]),
            Set([
                try XCTUnwrap(imported.bodiesByName["hinged"]),
                try XCTUnwrap(imported.bodiesByName["nested_without_joint"]),
            ]),
        ]))
    }

    func testDomainRandomizationIsSeededAndBounded() {
        let config = ArticulationDomainRandomization.conservativeSimToReal
        XCTAssertEqual(config.sample(seed: 42), config.sample(seed: 42))
        let sample = config.sample(seed: 43)
        XCTAssertTrue((0.90...1.10).contains(sample.mass))
        XCTAssertTrue((0.90...1.10).contains(sample.inertia))
        XCTAssertTrue((0.75...1.25).contains(sample.friction))
        XCTAssertTrue((0.85...1.05).contains(sample.motorTorque))
    }

    func testLinkAlignedInertiaMatchesUSDWithoutPrincipalAxes() throws {
        let xml = """
        <mujoco model="inertia-frame">
          <worldbody>
            <body name="root" pos="0 0 1">
              <inertial pos="0.2 0 0" quat="0.70710678 0 0 0.70710678"
                        mass="2" diaginertia="1 2 3"/>
            </body>
          </worldbody>
        </mujoco>
        """
        let asset = try MJCFAsset.parse(data: Data(xml.utf8))
        var principalScene = PhysicsScene(name: "principal")
        let principal = try asset.instantiate(
            in: &principalScene,
            options: MJCFInstantiationOptions())
        var linkScene = PhysicsScene(name: "link")
        let linkAligned = try asset.instantiate(
            in: &linkScene,
            options: MJCFInstantiationOptions(inertiaFrame: .linkAligned))

        let principalBody = principalScene.bodies[principal.rootBody]
        let linkBody = linkScene.bodies[linkAligned.rootBody]
        XCTAssertEqual(principalBody.position.x, 0.2, accuracy: 1e-6)
        XCTAssertEqual(linkBody.position.x, 0.2, accuracy: 1e-6)
        XCTAssertLessThan(abs(principalBody.rotation.real - 0.70710678), 1e-5)
        XCTAssertEqual(linkBody.rotation.real, 1, accuracy: 1e-6)
        XCTAssertLessThan(simd_length(linkBody.rotation.imag), 1e-6)
        XCTAssertEqual(
            linkAligned.linkFramesInBody["root"]!.position.x,
            -0.2, accuracy: 1e-6)
        XCTAssertEqual(
            linkAligned.linkFramesInBody["root"]!.rotation.real,
            1, accuracy: 1e-6)
    }
}
