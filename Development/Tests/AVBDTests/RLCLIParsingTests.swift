import Foundation
import XCTest

final class RLCLIParsingTests: XCTestCase {
    private struct Result {
        let status: Int32
        let stdout: String
        let stderr: String
    }

    private func runCLI(_ arguments: [String]) throws -> Result {
        let packageRoot = TestPaths.developmentPackage
        let executable = packageRoot
            .appendingPathComponent(".build/debug/avbd")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            XCTFail("missing avbd test executable at \(executable.path)")
            throw CocoaError(.fileNoSuchFile)
        }

        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = packageRoot
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        return Result(
            status: process.terminationStatus,
            stdout: String(
                decoding: stdout.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self),
            stderr: String(
                decoding: stderr.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self))
    }

    func testRegisteredRLCommandRejectsUnknownOption() throws {
        let result = try runCLI([
            "train-rl", "arachne15-velocity-v0", "--envz", "4",
        ])
        XCTAssertNotEqual(result.status, 0)
        XCTAssertEqual(result.stderr, "error: unknown option '--envz'\n")
    }

    func testRegisteredRLCommandRejectsKnownButIrrelevantOption() throws {
        let training = try runCLI([
            "train-rl", "arachne15-velocity-v0", "--frames", "4",
        ])
        XCTAssertNotEqual(training.status, 0)
        XCTAssertEqual(training.stderr, "error: unknown option '--frames'\n")

        let evaluation = try runCLI([
            "eval-rl", "arachne15-velocity-v0", "--updates", "1",
        ])
        XCTAssertNotEqual(evaluation.status, 0)
        XCTAssertEqual(evaluation.stderr, "error: unknown option '--updates'\n")

        let verification = try runCLI([
            "verify-policy-rl", "arachne15-goal-v0", "--lr", "0.001",
        ])
        XCTAssertNotEqual(verification.status, 0)
        XCTAssertEqual(verification.stderr, "error: unknown option '--lr'\n")
    }

    func testRegisteredRLCommandRejectsMalformedAndNonfiniteNumbers() throws {
        let malformed = try runCLI([
            "train-rl", "arachne15-velocity-v0", "--envs", "four",
        ])
        XCTAssertNotEqual(malformed.status, 0)
        XCTAssertEqual(
            malformed.stderr,
            "error: --envs expects an integer, got 'four'\n")

        let nonfinite = try runCLI([
            "train-rl", "arachne15-velocity-v0", "--lr", "nan",
        ])
        XCTAssertNotEqual(nonfinite.status, 0)
        XCTAssertEqual(
            nonfinite.stderr,
            "error: --lr expects a finite number, got 'nan'\n")
    }

    func testRegisteredRLCommandRejectsMissingValue() throws {
        let result = try runCLI([
            "eval-rl", "arachne15-velocity-v0", "--envs", "--episodes", "1",
        ])
        XCTAssertNotEqual(result.status, 0)
        XCTAssertEqual(result.stderr, "error: missing value after --envs\n")
    }

    func testTaskOptionAndValidNumericOptionsReachTaskLookup() throws {
        let result = try runCLI([
            "train-rl", "__strict_parser_missing_task__",
            "--envs", "2", "--seed", "42",
            "--task-option", "controlDecimation=4",
        ])
        XCTAssertNotEqual(result.status, 0)
        XCTAssertTrue(result.stderr.contains(
            "unknown RL task '__strict_parser_missing_task__'"))
        XCTAssertFalse(result.stderr.contains("unknown option"))
        XCTAssertFalse(result.stderr.contains("expects"))
    }

    func testDescribeRLAcceptsOnlyItsDeclaredJSONFlag() throws {
        let valid = try runCLI([
            "describe-rl", "arachne15-velocity-v0", "--json",
        ])
        XCTAssertEqual(valid.status, 0, valid.stderr)
        XCTAssertTrue(valid.stdout.contains("\"definitions\""))

        let typo = try runCLI([
            "describe-rl", "arachne15-velocity-v0", "--jsno",
        ])
        XCTAssertNotEqual(typo.status, 0)
        XCTAssertEqual(typo.stderr, "error: unknown option '--jsno'\n")
    }

    func testBooleanPresenceFlagRejectsTrailingValue() throws {
        let result = try runCLI([
            "train-rl", "arachne15-velocity-v0", "--resume", "false",
        ])
        XCTAssertNotEqual(result.status, 0)
        XCTAssertEqual(result.stderr, "error: unknown option 'false'\n")
    }

    func testTraceRejectsIgnoredTaskOptionsAndInvalidTrainingProgress() throws {
        let taskOption = try runCLI([
            "trace-rl", "__strict_parser_missing_task__",
            "--task-option", "trainingEnvironmentSteps=10",
        ])
        XCTAssertNotEqual(taskOption.status, 0)
        XCTAssertEqual(
            taskOption.stderr, "error: unknown option '--task-option'\n")

        let missingMode = try runCLI([
            "trace-rl", "__strict_parser_missing_task__",
            "--training-environment-steps", "10",
        ])
        XCTAssertNotEqual(missingMode.status, 0)
        XCTAssertEqual(
            missingMode.stderr,
            "error: --training-environment-steps requires --training-mode\n")

        let negative = try runCLI([
            "trace-rl", "__strict_parser_missing_task__", "--training-mode",
            "--training-environment-steps", "-1",
        ])
        XCTAssertNotEqual(negative.status, 0)
        XCTAssertEqual(
            negative.stderr,
            "error: --training-environment-steps must be non-negative\n")
    }

    func testEvidenceCommandsRejectUnknownMissingAndDuplicateOptions() throws {
        let typo = try runCLI(["select-rl", "--outpt", "selection.json"])
        XCTAssertNotEqual(typo.status, 0)
        XCTAssertEqual(typo.stderr, "error: unknown option '--outpt'\n")

        let missing = try runCLI([
            "aggregate-rl", "report.json", "--output",
        ])
        XCTAssertNotEqual(missing.status, 0)
        XCTAssertEqual(missing.stderr, "error: missing value after --output\n")

        let duplicate = try runCLI([
            "aggregate-checkpoint-rl", "report.json",
            "--output", "first.json", "--output", "second.json",
        ])
        XCTAssertNotEqual(duplicate.status, 0)
        XCTAssertEqual(
            duplicate.stderr,
            "error: duplicate --output for aggregate-checkpoint-rl\n")

        let verification = try runCLI([
            "verify-selection-rl", "selection.json", "--json",
        ])
        XCTAssertNotEqual(verification.status, 0)
        XCTAssertEqual(verification.stderr, "error: unknown option '--json'\n")
    }
}
