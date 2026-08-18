import Foundation
import simd
import XCTest
@testable import AVBDCore

final class UnitreeH1CollisionHullTests: XCTestCase {
    func testTaskRevisionBoundaryFollowsGeneratedHullTopology() throws {
        func make(_ configuration: HumanoidIsaacVelocityTaskConfig) throws
            -> HumanoidIsaacVelocityTask
        {
            try HumanoidIsaacVelocityTask(
                configuration: configuration,
                taskID: configuration.pointGoal
                    ? "humanoid-isaac-goal-v0"
                    : "humanoid-isaac-flat-v0")
        }

        let flat = try make(.init(numEnvironments: 1))
        let goal = try make(.init(
            numEnvironments: 1, pointGoal: true,
            projectileProbability: 0))
        XCTAssertEqual(flat.spec.revision,
                       RLPhysicsContract.deterministicColorSolveV1(11))
        XCTAssertEqual(goal.spec.revision,
                       RLPhysicsContract.deterministicColorSolveV1(3))
        XCTAssertEqual(enabledGeneratedHullCount(flat), 3)
        XCTAssertEqual(enabledGeneratedHullCount(goal), 3)

        let projectileConfigurations: [
            (HumanoidIsaacVelocityTaskConfig, Int)
        ] = [
            (.init(numEnvironments: 1, pointGoal: true,
                   projectileProbability: 1), 4),
            (.init(numEnvironments: 1, pointGoal: true,
                   projectileProbability: 1,
                   recoveryGatedActor: true), 5),
            (.init(numEnvironments: 1, pointGoal: true,
                   projectileProbability: 1,
                   recoveryGatedActor: true,
                   recoveryContextObservations: true), 6),
            (.init(numEnvironments: 1, pointGoal: true,
                   projectileProbability: 1,
                   recoveryGatedActor: true,
                   recoveryContextObservations: true,
                   recoveryExpertSide: -1), 7),
        ]
        for (configuration, revision) in projectileConfigurations {
            let task = try make(configuration)
            XCTAssertEqual(task.spec.revision,
                           RLPhysicsContract.deterministicColorSolveV1(revision))
            XCTAssertEqual(
                enabledGeneratedHullCount(task), 0,
                "projectile revision \(revision) uses full primitives, not hulls")
        }
    }

    func testGeneratedHullsFillReviewedMetalVertexBudget() {
        let hulls = [
            UnitreeH1CollisionHulls.leftAnkle,
            UnitreeH1CollisionHulls.rightAnkle,
            UnitreeH1CollisionHulls.torso,
        ]
        XCTAssertTrue(hulls.allSatisfy { $0.count == 64 })
        XCTAssertTrue(hulls.flatMap { $0 }.allSatisfy {
            $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
        })

        let leftBounds = bounds(UnitreeH1CollisionHulls.leftAnkle)
        let rightBounds = bounds(UnitreeH1CollisionHulls.rightAnkle)
        XCTAssertLessThan(simd_length(leftBounds.min - rightBounds.min), 1e-7)
        XCTAssertLessThan(simd_length(leftBounds.max - rightBounds.max), 1e-7)
        XCTAssertEqual(leftBounds.min.z, -0.07000001, accuracy: 1e-7)
        XCTAssertEqual(leftBounds.max.z, 0.010918009, accuracy: 1e-7)
    }

    func testGeneratedAnklesHaveMeasuredMirroredSupportEnvelope() {
        let directions = primitiveDirections(axisLimit: 8)
        var maximumDifference: Float = 0
        var squaredDifference: Double = 0
        for direction in directions {
            let reflected = F3(direction.x, -direction.y, direction.z)
            let left = support(
                UnitreeH1CollisionHulls.leftAnkle, direction: direction)
            let right = support(
                UnitreeH1CollisionHulls.rightAnkle, direction: reflected)
            let difference = abs(left - right)
            maximumDifference = max(maximumDifference, difference)
            squaredDifference += Double(difference * difference)
        }
        let rms = sqrt(squaredDifference / Double(directions.count))

        // The independently authored left/right source STLs are not exact
        // mirrors. Verify their generated physical envelopes stay within the
        // measured sub-millimetre contract instead of forcing fake symmetry.
        XCTAssertLessThanOrEqual(maximumDifference, 0.000926)
        XCTAssertLessThanOrEqual(rms, 0.000227)
    }

    func testPackagedProvenancePinsBSDSourceAndMeasuredErrors() throws {
        let url = try MJCFAsset.bundledResourceURL(
            resource: "COLLISION_HULLS_PROVENANCE", withExtension: "json",
            subdirectory: "Assets/unitree_h1")
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        let source = try XCTUnwrap(root["source"] as? [String: Any])
        XCTAssertEqual(source["license"] as? String, "BSD-3-Clause")
        XCTAssertEqual(
            source["revision"] as? String,
            "71f066ad0be9cd271f7ed58c030243ef157af9f4")

        let generator = try XCTUnwrap(root["generator"] as? [String: Any])
        XCTAssertEqual(generator["networkAccess"] as? Bool, false)
        XCTAssertEqual(generator["vertexLimit"] as? Int, 64)
        let probes = try XCTUnwrap(
            generator["probeDirections"] as? [String: Any])
        XCTAssertEqual(probes["count"] as? Int, 4_034)
        XCTAssertEqual(probes["antipodallySymmetric"] as? Bool, true)

        let output = try XCTUnwrap(root["output"] as? [String: Any])
        let meshes = try XCTUnwrap(output["meshes"] as? [[String: Any]])
        XCTAssertEqual(meshes.count, 3)
        XCTAssertTrue(meshes.allSatisfy {
            $0["generatedVertexCount"] as? Int == 64
        })
        XCTAssertTrue(meshes.allSatisfy {
            guard let digest = $0["sourceSHA256"] as? String else {
                return false
            }
            return digest.count == 64 && digest.allSatisfy { $0.isHexDigit }
        })

        let maximumErrors = try meshes.map {
            try XCTUnwrap($0["symmetricSupportErrorMaxMeters"] as? Double)
        }
        XCTAssertLessThanOrEqual(maximumErrors[0], 0.001251)
        XCTAssertLessThanOrEqual(maximumErrors[1], 0.001053)
        XCTAssertLessThanOrEqual(maximumErrors[2], 0.005259)

        XCTAssertNoThrow(try MJCFAsset.bundledResourceURL(
            resource: "LICENSE", withExtension: nil,
            subdirectory: "Assets/unitree_h1"))
    }

    private func bounds(_ vertices: [F3]) -> (min: F3, max: F3) {
        var minimum = F3(repeating: .greatestFiniteMagnitude)
        var maximum = F3(repeating: -.greatestFiniteMagnitude)
        for vertex in vertices {
            minimum = simd_min(minimum, vertex)
            maximum = simd_max(maximum, vertex)
        }
        return (minimum, maximum)
    }

    private func enabledGeneratedHullCount(
        _ task: HumanoidIsaacVelocityTask
    ) -> Int {
        task.environment.scene.colliders.filter {
            $0.collisionEnabled && $0.convexHullVertices.count == 64
        }.count
    }

    private func support(_ vertices: [F3], direction: F3) -> Float {
        vertices.map { simd_dot($0, direction) }.max()!
            / simd_length(direction)
    }

    private func primitiveDirections(axisLimit: Int) -> [F3] {
        var directions = [F3]()
        for x in -axisLimit...axisLimit {
            for y in -axisLimit...axisLimit {
                for z in -axisLimit...axisLimit where x != 0 || y != 0 || z != 0 {
                    if greatestCommonDivisor(
                        greatestCommonDivisor(abs(x), abs(y)), abs(z)) == 1 {
                        directions.append(F3(Float(x), Float(y), Float(z)))
                    }
                }
            }
        }
        return directions
    }

    private func greatestCommonDivisor(_ lhs: Int, _ rhs: Int) -> Int {
        var a = lhs
        var b = rhs
        while b != 0 {
            (a, b) = (b, a % b)
        }
        return a
    }
}
