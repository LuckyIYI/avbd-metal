import Foundation
import simd
import XCTest
@testable import AVBDCore

final class MJCFImporterTests: XCTestCase {
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
            defaultMotorGain: .init(stiffness: 100, damping: 5))
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
        var scene = PhysicsScene(name: "fixture")
        let imported = try asset.instantiate(
            in: &scene,
            defaultMotorGain: .init(stiffness: 20, damping: 3))

        XCTAssertEqual(scene.bodies.count, 2)
        XCTAssertEqual(scene.colliders.count, 2)
        XCTAssertEqual(scene.joints.count, 1)
        XCTAssertEqual(scene.bodies[imported.rootBody].position, F3(0.2, 0, 1))
        XCTAssertEqual(scene.colliders[0].localPosition, F3(-0.2, 0, 0.5))
        XCTAssertEqual(scene.colliders[0].size.x, 1, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].motorTorque, 7, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].motorDamping, 3, accuracy: 1e-6)
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
        let principal = try asset.instantiate(in: &principalScene)
        var linkScene = PhysicsScene(name: "link")
        let linkAligned = try asset.instantiate(
            in: &linkScene, inertiaFrame: .linkAligned)

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
