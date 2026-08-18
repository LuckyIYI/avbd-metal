#!/usr/bin/env python3
"""Hermetically verify every accepted Policy Replay result.

This check deliberately does not import MLX or execute a policy. It verifies the
immutable boundary around an already evaluated policy instead:

* accepted entries are discovered from ``PolicyReplayCatalog.swift``;
* tracked checkpoint identity is recomputed from the exact replay-semantic files;
* evaluation reports must match checkpoint task/revision/configuration/lineage;
* checkpoint robustness aggregates are rebuilt from their raw reports with the
  same Float32 quantile and pooling semantics as ``VectorPPO.swift``; and
* deployment-manifest hashes are checked when a checkpoint exposes them.

An accepted evidence shape that this verifier does not understand fails closed,
so extending the release catalog requires extending this contract as well.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EVALUATION_KEYS = {
    "task",
    "evaluationSeed",
    "episodes",
    "successes",
    "successRate",
    "taskMetrics",
}

H1_REQUALIFIED_SELECTION = "humanoid-isaac-flat-v1"
H1_TASK = "humanoid-isaac-flat-v0"
H1_SOURCE_REVISION = 1_000_010
H1_TARGET_REVISION = 1_000_011
H1_PARENT_DIRECTORY = "checkpoints/humanoid-isaac-flat-v0"
H1_CANDIDATE_DIRECTORY = (
    "runs/humanoid-isaac-flat-v1/requalification-r1000011/candidate"
)
H1_DECLARED_SOURCE_COMMIT = "d3d07bf1bd0b78c62235794a7c81621ffe748ed7"
H1_PARENT_FINGERPRINT = (
    "d6b5d416e7f7d75fa2b9b9dd33f78ae387e3f2a8139aa6d25a69e5dbcae777ab"
)
H1_CANDIDATE_FINGERPRINT = (
    "85571805cc7b688970cf5497beb5916be8fb3b1fcb7855207af6f55b208c7fd2"
)
H1_POLICY_SHA256 = (
    "3e0a21600afd6ee2e50383ed33f69007f9855c3f8b85ec0b52c4f2acc2c285ae"
)
H1_REQUALIFICATION_MANIFEST_SHA256 = (
    "01628ae5d07d75a4538cbcf085b27314b2e72e4c5324f6bd4249d6297e03da03"
)
H1_AGGREGATE_SHA256 = (
    "a3e54308ac978e68509ff1ea437a7f908be3b2c2a0df6699755e03a98d7e3f6d"
)
H1_REPORT_RESULTS = {
    51_001: (506,
             "2c48696385668a22417bfb2d811adb39c05e6a501ac3ef7bf74bef2d046d3815"),
    51_002: (507,
             "e0c4c03dcbfa18b6258f87ed135bf70304e97bc4640a6d4858026513ea57aba9"),
    51_003: (505,
             "7412147895ddb1a48eec036a7e9d79bdd2379139b4997e62a6abbddf5e9fb124"),
    51_004: (510,
             "015bbb14ec3c9d1c7462ed1cd584f8af84685a02f0818f43428df6d6559b9e60"),
}
H1_TOTAL_SUCCESSES = 2_028
H1_EVALUATION_SEEDS = [51_001, 51_002, 51_003, 51_004]
H1_EVALUATION_ENVIRONMENTS = 128
H1_EPISODES_PER_REPORT = 512
H1_TASK_CONFIGURATION = {
    "commandResamplingSteps": 500,
    "initialYawRange": 3.1415925,
    "observationNoise": 1,
    "solverIterations": 20,
    "standingCommandProbability": 0.02,
}
H1_EVALUATION_CRITERIA = {
    "minimumSuccessRate": 0.8,
    "minimumMeanEpisodeLengthFraction": 0.9,
    "minimumTaskMetrics": {"episode/survived": 0.9},
    "maximumTaskMetrics": {
        "episode/linear_velocity_rmse_mps": 0.35,
        "episode/yaw_rate_rmse_rps": 0.5,
    },
}
REQUALIFICATION_CHANGED_FIELDS = [
    "metadata.taskRevision",
    "metadata.inferenceBatchSize",
    "metadata.ppo.initializationCheckpoint",
    "metadata.ppo.updates",
    "training-state.completedUpdates",
    "training-state.environmentSteps",
    "training-state.optimizerSteps",
    "training-state.adaptiveLearningRate",
]
REQUALIFICATION_MANIFEST_KEYS = {
    "schemaVersion", "task", "sourceTaskRevision", "targetTaskRevision",
    "parentCheckpointDirectory", "candidateCheckpointDirectory",
    "parentCheckpointFingerprint", "candidateCheckpointFingerprint",
    "parentPolicySHA256", "candidatePolicySHA256", "parentMetadataSHA256",
    "candidateMetadataSHA256", "parentTrainingStateSHA256",
    "candidateTrainingStateSHA256", "declaredSourceCommit",
    "taskConfiguration", "observationDimension", "actionDimension",
    "simulationStepSeconds", "controlDecimation", "maxEpisodeSteps",
    "inferenceBatchSize", "parentTrainingUpdates",
    "parentTrainingEnvironmentSteps", "targetTrainingUpdates",
    "targetTrainingEnvironmentSteps", "changedFields", "qualificationPlan",
    "evaluationCriteria", "qualification",
}


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    selection_id: str
    task_id: str
    runtime: str
    checkpoint_relative_directory: str
    evidence_relative_path: str


@dataclass(frozen=True)
class CheckpointContext:
    entry: CatalogEntry
    directory: Path
    metadata: dict[str, Any]
    training_state: dict[str, Any]
    fingerprint: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_exact_keys(value: Any, keys: set[str], label: str) -> None:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    require(value.keys() == keys,
            f"{label}: keys changed; expected {sorted(keys)}, "
            f"got {sorted(value)}")


def reject_nonfinite(token: str) -> None:
    raise VerificationError(f"JSON contains non-finite number {token}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def repository_path(root: Path, relative: str, label: str) -> Path:
    require(relative and not Path(relative).is_absolute(),
            f"{label} must be a non-empty repository-relative path")
    path = (root / relative).resolve()
    require(path == root or root in path.parents,
            f"{label} escapes the repository: {relative}")
    return path


def sealed_bundle_file(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative,
            f"{label} must be a non-empty relative path")
    requested = Path(relative)
    require(not requested.is_absolute()
            and all(part not in {"", ".", ".."} for part in requested.parts),
            f"{label} is not a canonical bundle-relative path: {relative!r}")

    # Git should contain evidence bytes, not aliases to mutable files. Resolve
    # for containment and independently reject every symlink component.
    cursor = root
    for part in requested.parts:
        cursor /= part
        require(not cursor.is_symlink(), f"{label} must not contain a symlink")
    resolved_root = root.resolve()
    resolved = cursor.resolve()
    require(resolved_root in resolved.parents,
            f"{label} escapes its checkpoint bundle: {relative}")
    require(resolved.is_file(), f"missing {label}: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing hashed file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(directory: Path) -> str:
    """Match VectorPPOTrainer.checkpointFingerprint exactly."""
    digest = hashlib.sha256()
    for name in ("metadata.json", "policy.safetensors", "training-state.json"):
        path = directory / name
        require(path.is_file(), f"incomplete checkpoint; missing {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def f32(value: int | float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_bits(value: int | float) -> bytes:
    return struct.pack("<f", f32(value))


def config_equal(first: Any, second: Any) -> bool:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first.keys() != second.keys():
        return False
    for key in first:
        a, b = first[key], second[key]
        if isinstance(a, bool) or isinstance(b, bool):
            if type(a) is not type(b) or a != b:
                return False
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isfinite(float(a)) or not math.isfinite(float(b)):
                return False
            if f32_bits(a) != f32_bits(b):
                return False
        elif a != b:
            return False
    return True


def assert_json_equal(actual: Any, expected: Any, label: str) -> None:
    """Compare a reconstructed aggregate with Swift Float32 semantics."""
    if expected is None or isinstance(expected, (str, bool)):
        require(type(actual) is type(expected) and actual == expected,
                f"{label}: expected {expected!r}, got {actual!r}")
        return
    if type(expected) is int:
        require(type(actual) is int and actual == expected,
                f"{label}: expected integer {expected}, got {actual!r}")
        return
    if isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool),
                f"{label}: expected Float32, got {actual!r}")
        require(math.isfinite(float(actual)), f"{label}: non-finite value")
        require(f32_bits(actual) == f32_bits(expected),
                f"{label}: expected Float32 {expected!r}, got {actual!r}")
        return
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f"{label}: list shape changed")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_json_equal(actual_item, expected_item, f"{label}[{index}]")
        return
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label}: expected object")
        require(actual.keys() == expected.keys(),
                f"{label}: keys changed; expected {sorted(expected)}, "
                f"got {sorted(actual)}")
        for key in expected:
            assert_json_equal(actual[key], expected[key], f"{label}.{key}")
        return
    raise VerificationError(f"{label}: unsupported reconstructed value {expected!r}")


def matching_delimiter(source: str, start: int, opening: str, closing: str) -> int:
    require(source[start] == opening, "internal delimiter parser error")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        character = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    raise VerificationError(f"unterminated {opening} in PolicyReplayCatalog")


def argument_expression(entry: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*:\s*", entry)
    require(match is not None, f"catalog entry is missing {name}")
    start = match.end()
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    in_string = False
    escaped = False
    for index in range(start, len(entry)):
        character = entry[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in pairs:
            stack.append(pairs[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character == "," and not stack:
            return entry[start:index].strip()
    return entry[start:].rstrip().rstrip(")").strip()


def swift_string(expression: str, label: str) -> str | None:
    if expression == "nil":
        return None
    literals = list(re.finditer(r'"(?:\\.|[^"\\])*"', expression))
    require(bool(literals), f"{label} is not a static Swift string")
    remainder = expression
    for match in reversed(literals):
        remainder = remainder[:match.start()] + remainder[match.end():]
    require(re.fullmatch(r"[+\s]*", remainder) is not None,
            f"{label} must use only static string literals")
    try:
        return "".join(json.loads(match.group(0)) for match in literals)
    except json.JSONDecodeError as error:
        raise VerificationError(f"cannot decode {label}: {error}") from error


def catalog_entries(root: Path) -> list[CatalogEntry]:
    path = root / "Sources/AVBDLearn/PolicyReplayCatalog.swift"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    declaration = source.find("public static let entries")
    require(declaration >= 0, "PolicyReplayCatalog.entries is missing")
    assignment = source.find("=", declaration)
    array_start = source.find("[", assignment)
    require(assignment >= 0 and array_start >= 0,
            "PolicyReplayCatalog.entries is not a literal array")
    array_end = matching_delimiter(source, array_start, "[", "]")
    body = source[array_start + 1:array_end]

    blocks: list[str] = []
    cursor = 0
    while True:
        match = re.search(r"\.init\s*\(", body[cursor:])
        if match is None:
            break
        open_paren = cursor + match.end() - 1
        close_paren = matching_delimiter(body, open_paren, "(", ")")
        blocks.append(body[open_paren + 1:close_paren])
        cursor = close_paren + 1
    require(bool(blocks), "PolicyReplayCatalog contains no literal entries")

    accepted: list[CatalogEntry] = []
    all_selection_ids: set[str] = set()
    for block in blocks:
        selection_id = swift_string(
            argument_expression(block, "selectionID"), "selectionID")
        require(selection_id is not None and selection_id not in all_selection_ids,
                f"invalid or duplicate Policy Replay selection {selection_id!r}")
        all_selection_ids.add(selection_id)
        qualification = argument_expression(block, "qualification")
        if qualification != ".accepted":
            continue
        task_id = swift_string(argument_expression(block, "taskID"), "taskID")
        checkpoint = swift_string(
            argument_expression(block, "checkpointRelativeDirectory"),
            "checkpointRelativeDirectory",
        )
        evidence = swift_string(
            argument_expression(block, "evidenceRelativePath"),
            "evidenceRelativePath",
        )
        runtime = argument_expression(block, "runtime")
        require(task_id is not None and checkpoint is not None and evidence is not None,
                f"accepted selection {selection_id} must name task, checkpoint and evidence")
        require(runtime == ".nativeMLX",
                f"accepted selection {selection_id} uses unsupported runtime {runtime}")
        accepted.append(CatalogEntry(
            selection_id=selection_id,
            task_id=task_id,
            runtime=runtime,
            checkpoint_relative_directory=checkpoint,
            evidence_relative_path=evidence,
        ))
    require(bool(accepted), "PolicyReplayCatalog contains no accepted policies")
    return accepted


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def verify_deployment_manifest(context: CheckpointContext,
                               required: bool = False,
                               exact: bool = False) -> None:
    path = context.directory / "deployment-manifest.json"
    if not path.exists():
        require(not required, f"{path}: required deployment manifest is missing")
        return
    manifest = load_json(path)
    metadata = context.metadata
    state = context.training_state
    require(manifest.get("schemaVersion") == 1,
            f"{path}: unsupported deployment schema")
    require(manifest.get("metadataFile") == "metadata.json"
            and manifest.get("policyFile") == "policy.safetensors"
            and manifest.get("trainingStateFile") == "training-state.json",
            f"{path}: deployment file mapping changed")
    require(manifest.get("checkpointFingerprint") == context.fingerprint,
            f"{path}: checkpoint fingerprint mismatch")
    policy_hash = require_sha256(manifest.get("policySHA256"),
                                 f"{path}: policySHA256")
    require(sha256_file(context.directory / "policy.safetensors") == policy_hash,
            f"{path}: policy SHA-256 mismatch")
    require(manifest.get("task") == metadata.get("task")
            and manifest.get("taskRevision") == metadata.get("taskRevision"),
            f"{path}: task identity differs from checkpoint metadata")
    require(config_equal(manifest.get("taskConfiguration"),
                         metadata.get("taskConfiguration")),
            f"{path}: task configuration differs from checkpoint metadata")
    for field in ("architectureVersion", "observationDimension", "actionDimension",
                  "controlDecimation"):
        require(manifest.get(field) == metadata.get(field),
                f"{path}: {field} differs from checkpoint metadata")
    require(f32_bits(manifest.get("simulationStepSeconds"))
            == f32_bits(metadata.get("simulationStep")),
            f"{path}: simulation step differs from checkpoint metadata")
    require(manifest.get("trainingUpdates") == state.get("completedUpdates")
            and manifest.get("trainingEnvironmentSteps") == state.get("environmentSteps"),
            f"{path}: training progress differs from checkpoint state")

    if exact:
        ppo = metadata.get("ppo")
        require(isinstance(ppo, dict),
                f"{path}: checkpoint PPO metadata is missing")
        normalize = ppo.get("normalizeObservations")
        distribution = ppo.get("actionDistribution")
        require(type(normalize) is bool,
                f"{path}: observation normalization mode is missing")
        require(isinstance(distribution, str) and distribution,
                f"{path}: action distribution is missing")
        step = metadata.get("simulationStep")
        decimation = metadata.get("controlDecimation")
        require(isinstance(step, (int, float)) and not isinstance(step, bool)
                and type(decimation) is int and decimation > 0,
                f"{path}: control timing metadata is invalid")
        frequency = f32(f32(1) / f32(f32(step) * f32(decimation)))
        expected: dict[str, Any] = {
            "schemaVersion": 1,
            "task": metadata["task"],
            "taskRevision": metadata["taskRevision"],
            "checkpointFingerprint": context.fingerprint,
            "policySHA256": sha256_file(context.directory / "policy.safetensors"),
            "policyFile": "policy.safetensors",
            "metadataFile": "metadata.json",
            "trainingStateFile": "training-state.json",
            "architectureVersion": metadata["architectureVersion"],
            "observationDimension": metadata["observationDimension"],
            "actionDimension": metadata["actionDimension"],
            "simulationStepSeconds": metadata["simulationStep"],
            "controlDecimation": decimation,
            "controlFrequencyHz": frequency,
            "normalizesObservations": normalize,
            "actionDistribution": distribution,
            "taskConfiguration": metadata["taskConfiguration"],
            "trainingUpdates": state["completedUpdates"],
            "trainingEnvironmentSteps": state["environmentSteps"],
        }
        if normalize:
            expected["observationNormalizationClip"] = 10.0
        assert_json_equal(manifest, expected, str(path))


def verify_checkpoint(root: Path, entry: CatalogEntry) -> CheckpointContext:
    directory = repository_path(
        root, f"checkpoints/{entry.checkpoint_relative_directory}",
        f"{entry.selection_id} checkpoint",
    )
    require(directory.is_dir(),
            f"{entry.selection_id}: missing checkpoint directory {directory}")
    metadata = load_json(directory / "metadata.json")
    state = load_json(directory / "training-state.json")
    require(metadata.get("task") == entry.task_id,
            f"{entry.selection_id}: checkpoint task differs from catalog")
    require(type(metadata.get("taskRevision")) is int,
            f"{entry.selection_id}: checkpoint taskRevision is missing")
    require(isinstance(metadata.get("taskConfiguration"), dict),
            f"{entry.selection_id}: checkpoint task configuration is missing")
    require(type(state.get("completedUpdates")) is int
            and type(state.get("environmentSteps")) is int,
            f"{entry.selection_id}: checkpoint training state is incomplete")
    fingerprint = checkpoint_fingerprint(directory)
    context = CheckpointContext(entry, directory, metadata, state, fingerprint)
    verify_deployment_manifest(context)
    return context


def task_acceptance_failures(report: dict[str, Any],
                             context: CheckpointContext) -> list[str]:
    """Re-evaluate the task-owned publication gate for accepted revisions."""
    task = context.entry.task_id
    revision = context.metadata["taskRevision"]
    configuration = report["evaluationTaskConfiguration"]
    metrics = report["taskMetrics"]
    failures: list[str] = []

    def minimum(name: str, threshold: float) -> None:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            failures.append(f"missing metric {name}")
        elif f32(value) < f32(threshold):
            failures.append(f"{name} below threshold")

    def maximum(name: str, threshold: float) -> None:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            failures.append(f"missing metric {name}")
        elif f32(value) > f32(threshold):
            failures.append(f"{name} above threshold")

    if task == H1_TASK and revision == H1_TARGET_REVISION:
        require(context.entry.selection_id == H1_REQUALIFIED_SELECTION,
                "H1 revision 1000011 evidence must use the requalified selection")
        require(config_equal(configuration, H1_TASK_CONFIGURATION),
                "accepted H1 evidence changed the task configuration")
        if f32(report["successRate"]) < f32(0.80):
            failures.append("success rate below threshold")
        minimum("episode/survived", 0.90)
        maximum("episode/linear_velocity_rmse_mps", 0.35)
        maximum("episode/yaw_rate_rmse_rps", 0.50)
        minimum_length_fraction = f32(0.90)
    else:
        require(revision == 6
                and task in {"arachne15-goal-v0", "arachne15-velocity-v0"},
                f"{context.entry.selection_id}: accepted task criteria are not verified")

    if task == "arachne15-goal-v0":
        require(f32_bits(configuration.get("pointGoal")) == f32_bits(1),
                "accepted Arachne Goal evidence disabled pointGoal")
        if f32(report["successRate"]) < f32(0.90):
            failures.append("success rate below threshold")
        minimum("episode/survived", 0.95)
        minimum("episode/goal_reached", 0.90)
        minimum("episode/goal_front_success_rate", 0.85)
        minimum("episode/goal_near_success_rate", 0.85)
        minimum("episode/goal_far_success_rate", 0.85)
        minimum("episode/minimum_foot_collider_clearance_m", -0.003)
        direction = f32(configuration["maximumGoalDirectionAngle"])
        if direction > f32(f32(math.pi) / f32(4)):
            minimum("episode/goal_left_success_rate", 0.85)
            minimum("episode/goal_right_success_rate", 0.85)
        if direction > f32(f32(3) * f32(math.pi) / f32(4)):
            minimum("episode/goal_rear_success_rate", 0.85)
        radius = f32(configuration["goalRadius"])
        maximum("episode/final_goal_distance_m", f32(radius * f32(1.25)))
        maximum("episode/minimum_goal_distance_m", radius)
        maximum("episode/yaw_rate_rmse_rps", 0.40)
        maximum("episode/foot_collider_penetration_rmse_m", 0.0005)
        maximum("episode/foot_collider_penetration_over_1mm_fraction", 0.025)
        minimum_length_fraction = f32(0.10)
    elif task == "arachne15-velocity-v0":
        require(f32_bits(configuration.get("pointGoal")) == f32_bits(0),
                "accepted Arachne Velocity evidence enabled pointGoal")
        if f32(report["successRate"]) < f32(0.80):
            failures.append("success rate below threshold")
        minimum("episode/survived", 0.90)
        maximum("episode/linear_velocity_rmse_mps", 0.15)
        maximum("episode/yaw_rate_rmse_rps", 0.40)
        minimum_length_fraction = f32(0.90)

    max_episode_steps = context.metadata.get("maxEpisodeSteps")
    require(type(max_episode_steps) is int and max_episode_steps > 0,
            f"{context.entry.selection_id}: maxEpisodeSteps is missing")
    minimum_length = f32(minimum_length_fraction * f32(max_episode_steps))
    if f32(report["meanEpisodeLength"]) < minimum_length:
        failures.append("mean episode length below threshold")
    return failures


def require_evaluation_report(report: dict[str, Any], context: CheckpointContext,
                              label: str, minimum_episodes: int) -> None:
    metadata = context.metadata
    state = context.training_state
    require(type(report.get("provenanceVersion")) is int
            and report["provenanceVersion"] >= 3,
            f"{label}: accepted evidence requires provenanceVersion >= 3")
    require(report.get("task") == context.entry.task_id,
            f"{label}: task differs from catalog")
    require(type(report.get("taskRevision")) is int,
            f"{label}: taskRevision is missing")
    require(report["taskRevision"] == metadata["taskRevision"],
            f"{label}: taskRevision differs from checkpoint")
    report_fingerprint = require_sha256(
        report.get("checkpointFingerprint"), f"{label}: checkpointFingerprint")
    require(report_fingerprint == context.fingerprint,
            f"{label}: checkpoint fingerprint mismatch")
    require(config_equal(report.get("checkpointTaskConfiguration"),
                         metadata.get("taskConfiguration")),
            f"{label}: checkpoint task configuration mismatch")
    evaluation_configuration = report.get("evaluationTaskConfiguration")
    require(isinstance(evaluation_configuration, dict),
            f"{label}: evaluation task configuration is missing")
    transferred = report.get("taskConfigurationTransferred")
    require(type(transferred) is bool,
            f"{label}: taskConfigurationTransferred is missing")
    require(transferred == (not config_equal(
        report["checkpointTaskConfiguration"], evaluation_configuration)),
        f"{label}: task transfer flag does not describe its configurations")

    ppo = metadata.get("ppo")
    require(isinstance(ppo, dict), f"{label}: checkpoint PPO metadata is missing")
    require(report.get("trainingSeed") == ppo.get("seed"),
            f"{label}: training seed differs from checkpoint metadata")
    require(report.get("initializationCheckpoint")
            == ppo.get("initializationCheckpoint"),
            f"{label}: initialization lineage differs from checkpoint metadata")
    require(report.get("trainingUpdates") == state.get("completedUpdates")
            and report.get("trainingEnvironmentSteps") == state.get("environmentSteps"),
            f"{label}: training progress differs from checkpoint state")
    require(isinstance(report.get("checkpointDirectory"), str)
            and bool(report["checkpointDirectory"]),
            f"{label}: source checkpoint directory is missing")
    require(type(report.get("evaluationSeed")) is int,
            f"{label}: evaluation seed is missing")
    require(type(report.get("evaluationEnvironments")) is int
            and report["evaluationEnvironments"] > 0,
            f"{label}: evaluation replica count is missing")

    episodes = report.get("episodes")
    successes = report.get("successes")
    require(type(episodes) is int and episodes >= minimum_episodes,
            f"{label}: requires at least {minimum_episodes} episodes")
    require(type(successes) is int and 0 <= successes <= episodes,
            f"{label}: invalid success count")
    expected_rate = f32(f32(successes) / f32(episodes))
    require(isinstance(report.get("successRate"), (int, float))
            and f32_bits(report["successRate"]) == f32_bits(expected_rate),
            f"{label}: successRate does not equal successes / episodes")
    for field in ("successRate", "meanReturn", "meanEpisodeLength"):
        value = report.get(field)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"{label}: {field} is missing or non-finite")
    metrics = report.get("taskMetrics")
    require(isinstance(metrics, dict) and bool(metrics),
            f"{label}: task metrics are missing")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) for value in metrics.values()),
            f"{label}: task metrics contain a non-finite value")
    acceptance = report.get("acceptance")
    require(isinstance(acceptance, dict)
            and acceptance.get("passed") is True
            and acceptance.get("failures") == [],
            f"{label}: accepted evidence did not pass its task-owned gate")
    failures = task_acceptance_failures(report, context)
    require(not failures,
            f"{label}: task-owned acceptance recomputation failed: "
            + "; ".join(failures))


def float32_quantile(sorted_values: list[float], probability: float) -> float:
    index = f32(f32(probability) * f32(len(sorted_values) - 1))
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = f32(index - f32(lower))
    first = f32(sorted_values[lower] * f32(f32(1) - fraction))
    second = f32(sorted_values[upper] * fraction)
    return f32(first + second)


def summarize(values: list[int | float]) -> dict[str, float]:
    require(bool(values), "cannot summarize an empty metric")
    ordered = sorted(f32(value) for value in values)
    return {
        "median": float32_quantile(ordered, 0.5),
        "firstQuartile": float32_quantile(ordered, 0.25),
        "thirdQuartile": float32_quantile(ordered, 0.75),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def reports_share_checkpoint(first: dict[str, Any], report: dict[str, Any]) -> bool:
    scalar_fields = (
        "task", "taskRevision", "checkpointDirectory", "checkpointFingerprint",
        "initializationCheckpoint", "trainingSeed", "trainingUpdates",
        "trainingEnvironmentSteps", "evaluationEnvironments",
        "taskConfigurationTransferred",
    )
    return all(report.get(field) == first.get(field) for field in scalar_fields) \
        and config_equal(report.get("checkpointTaskConfiguration"),
                         first.get("checkpointTaskConfiguration")) \
        and config_equal(report.get("evaluationTaskConfiguration"),
                         first.get("evaluationTaskConfiguration"))


def rebuild_checkpoint_aggregate(reports: list[dict[str, Any]],
                                 required_runs: int,
                                 required_episodes: int) -> dict[str, Any]:
    require(required_runs == 4 and required_episodes == 512,
            "accepted checkpoint qualification must retain the 4 x 512 contract")
    require(bool(reports), "checkpoint aggregate has no raw reports")
    first = reports[0]
    require(type(first.get("taskRevision")) is int
            and all(type(report.get("taskRevision")) is int for report in reports),
            "checkpoint robustness reports require an explicit task revision")
    require(all(reports_share_checkpoint(first, report) for report in reports),
            "checkpoint robustness reports do not describe one immutable checkpoint")
    seeds = [report.get("evaluationSeed") for report in reports]
    require(all(type(seed) is int for seed in seeds)
            and len(set(seeds)) == len(seeds),
            "checkpoint robustness reports require distinct evaluation seeds")

    metric_names = set().union(*(report["taskMetrics"].keys() for report in reports))
    require(all(all(name in report["taskMetrics"] for name in metric_names)
                for report in reports),
            "a task metric is missing from one or more reports")
    total_episodes = sum(report["episodes"] for report in reports)
    total_successes = sum(report["successes"] for report in reports)
    require(total_episodes > 0, "checkpoint aggregate has no evaluated episodes")
    accepted = sum(report.get("acceptance", {}).get("passed") is True
                   for report in reports)
    has_required_runs = len(reports) >= required_runs
    has_required_episodes = all(
        report["episodes"] >= required_episodes for report in reports
    )
    all_passed = accepted == len(reports)
    provenance_complete = all(
        (report.get("provenanceVersion") or 0) >= 2
        and type(report.get("taskRevision")) is int
        and (report.get("evaluationEnvironments") or 0) > 0
        and bool(report.get("checkpointFingerprint"))
        and ((report.get("provenanceVersion") or 0) < 3
             or (report.get("checkpointTaskConfiguration") is not None
                 and report.get("evaluationTaskConfiguration") is not None
                 and report.get("taskConfigurationTransferred") is not None))
        for report in reports
    )
    expected: dict[str, Any] = {
        "scope": "single_checkpoint_across_evaluation_seeds",
        "task": first["task"],
        "taskRevision": first["taskRevision"],
        "evaluationTaskConfiguration": first.get("evaluationTaskConfiguration"),
        "taskConfigurationTransferred": first.get("taskConfigurationTransferred"),
        "checkpointDirectory": first["checkpointDirectory"],
        "trainingSeed": first["trainingSeed"],
        "checkpointFingerprint": first.get("checkpointFingerprint"),
        "initializationCheckpoint": first.get("initializationCheckpoint"),
        "evaluationSeeds": sorted(seeds),
        "evaluationEnvironments": first.get("evaluationEnvironments"),
        "runs": len(reports),
        "requiredRuns": required_runs,
        "requiredEpisodesPerRun": required_episodes,
        "totalEpisodes": total_episodes,
        "totalSuccesses": total_successes,
        "pooledSuccessRate": f32(f32(total_successes) / f32(total_episodes)),
        "acceptedRuns": accepted,
        "allRunsPassed": all_passed,
        "hasRequiredRunCount": has_required_runs,
        "allRunsHaveRequiredEpisodes": has_required_episodes,
        "provenanceComplete": provenance_complete,
        "robustAcrossEvaluationSeeds": (
            has_required_runs and has_required_episodes
            and all_passed and provenance_complete
        ),
        "successRate": summarize([report["successRate"] for report in reports]),
        "meanReturn": summarize([report["meanReturn"] for report in reports]),
        "meanEpisodeLength": summarize(
            [report["meanEpisodeLength"] for report in reports]
        ),
        "taskMetrics": {
            name: summarize([report["taskMetrics"][name] for report in reports])
            for name in metric_names
        },
    }
    # Synthesized Swift Codable omits nil optionals.
    return {key: value for key, value in expected.items() if value is not None}


def aggregate_reports(path: Path, aggregate: dict[str, Any]) \
        -> list[tuple[Path, dict[str, Any]]]:
    seeds = aggregate.get("evaluationSeeds")
    require(isinstance(seeds, list) and all(type(seed) is int for seed in seeds),
            f"{path}: evaluationSeeds is missing")
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for candidate_path in sorted(path.parent.glob("*.json")):
        candidate = load_json(candidate_path)
        if EVALUATION_KEYS.issubset(candidate):
            candidates.append((candidate_path, candidate))

    selected: list[tuple[Path, dict[str, Any]]] = []
    for seed in seeds:
        matches = [item for item in candidates if item[1].get("evaluationSeed") == seed
                   and item[1].get("task") == aggregate.get("task")
                   and item[1].get("taskRevision") == aggregate.get("taskRevision")
                   and item[1].get("checkpointDirectory")
                       == aggregate.get("checkpointDirectory")
                   and item[1].get("checkpointFingerprint")
                       == aggregate.get("checkpointFingerprint")
                   and item[1].get("initializationCheckpoint")
                       == aggregate.get("initializationCheckpoint")
                   and item[1].get("trainingSeed") == aggregate.get("trainingSeed")
                   and item[1].get("evaluationEnvironments")
                       == aggregate.get("evaluationEnvironments")
                   and item[1].get("taskConfigurationTransferred")
                       == aggregate.get("taskConfigurationTransferred")
                   and config_equal(item[1].get("evaluationTaskConfiguration"),
                                    aggregate.get("evaluationTaskConfiguration"))]
        require(len(matches) == 1,
                f"{path}: expected exactly one raw report for seed {seed}, "
                f"found {len(matches)}")
        selected.append(matches[0])
    return selected


def verify_aggregate(path: Path, context: CheckpointContext) -> int:
    aggregate = load_json(path)
    require(aggregate.get("scope") == "single_checkpoint_across_evaluation_seeds",
            f"{path}: unsupported aggregate scope")
    require(type(aggregate.get("taskRevision")) is int,
            f"{path}: taskRevision is missing")
    require(aggregate.get("task") == context.entry.task_id
            and aggregate["taskRevision"] == context.metadata["taskRevision"],
            f"{path}: aggregate task identity differs from checkpoint")
    require(aggregate.get("checkpointFingerprint") == context.fingerprint,
            f"{path}: aggregate checkpoint fingerprint mismatch")
    required_runs = aggregate.get("requiredRuns")
    required_episodes = aggregate.get("requiredEpisodesPerRun")
    require(type(required_runs) is int and type(required_episodes) is int,
            f"{path}: qualification requirements are missing")
    reports = aggregate_reports(path, aggregate)
    require(len(reports) == aggregate.get("runs"),
            f"{path}: aggregate run count differs from its raw reports")
    for report_path, report in reports:
        require_evaluation_report(
            report, context, str(report_path), required_episodes
        )
    rebuilt = rebuild_checkpoint_aggregate(
        [report for _, report in reports], required_runs, required_episodes
    )
    assert_json_equal(aggregate, rebuilt, str(path))
    require(aggregate.get("robustAcrossEvaluationSeeds") is True,
            f"{path}: accepted checkpoint is not robust across evaluation seeds")
    return len(reports)


def verify_h1_requalification(root: Path, context: CheckpointContext,
                              evidence_path: Path) -> Path:
    """Verify the sealed, zero-update H1 revision transfer.

    This is an integrity check for evidence produced by a trusted operator, not
    a signature or remote attestation. It deliberately does not merely trust
    the manifest's digests: the permitted candidate metadata/state transform is
    re-derived from the immutable parent and every task-owned threshold and
    qualification seed is fixed here as release policy.
    """
    require(context.entry.selection_id == H1_REQUALIFIED_SELECTION,
            "internal H1 requalification selection mismatch")
    require(context.entry.task_id == H1_TASK,
            f"{context.entry.selection_id}: catalog task changed")
    require(context.entry.checkpoint_relative_directory
            == H1_REQUALIFIED_SELECTION,
            f"{context.entry.selection_id}: checkpoint directory changed")
    require(context.entry.evidence_relative_path
            == ("checkpoints/humanoid-isaac-flat-v1/"
                "requalification-manifest.json"),
            f"{context.entry.selection_id}: evidence path changed")
    require(context.metadata.get("taskRevision") == H1_TARGET_REVISION,
            f"{context.entry.selection_id}: target task revision must be "
            f"{H1_TARGET_REVISION}")

    manifest_path = context.directory / "requalification-manifest.json"
    expected_top_level = {
        "metadata.json", "policy.safetensors", "training-state.json",
        "deployment-manifest.json", "requalification-manifest.json",
        "qualification",
    }
    require({path.name for path in context.directory.iterdir()}
            == expected_top_level,
            f"{context.directory}: release bundle has missing or extra files")
    require(not manifest_path.is_symlink(),
            f"{manifest_path}: manifest must not be a symlink")
    require(sha256_file(manifest_path) == H1_REQUALIFICATION_MANIFEST_SHA256,
            f"{manifest_path}: pinned release manifest bytes changed")
    require(evidence_path.resolve() == manifest_path.resolve(),
            f"{context.entry.selection_id}: accepted evidence must name the "
            "sealed requalification manifest")
    manifest = load_json(manifest_path)
    require_exact_keys(
        manifest, REQUALIFICATION_MANIFEST_KEYS, str(manifest_path))
    require(type(manifest.get("schemaVersion")) is int
            and manifest["schemaVersion"] == 1,
            f"{manifest_path}: unsupported requalification schema")
    require(manifest.get("task") == H1_TASK
            and type(manifest.get("sourceTaskRevision")) is int
            and manifest["sourceTaskRevision"] == H1_SOURCE_REVISION
            and type(manifest.get("targetTaskRevision")) is int
            and manifest["targetTaskRevision"] == H1_TARGET_REVISION,
            f"{manifest_path}: H1 revision lineage changed")
    require(manifest.get("parentCheckpointDirectory") == H1_PARENT_DIRECTORY,
            f"{manifest_path}: parent checkpoint changed")
    require(manifest.get("candidateCheckpointDirectory")
            == H1_CANDIDATE_DIRECTORY,
            f"{manifest_path}: evaluated candidate path changed")
    source_commit = manifest.get("declaredSourceCommit")
    require(source_commit == H1_DECLARED_SOURCE_COMMIT,
            f"{manifest_path}: declared source commit changed")
    require(manifest.get("changedFields") == REQUALIFICATION_CHANGED_FIELDS,
            f"{manifest_path}: permitted zero-update transform changed")
    assert_json_equal(
        manifest.get("taskConfiguration"), H1_TASK_CONFIGURATION,
        f"{manifest_path}.taskConfiguration")
    assert_json_equal(
        manifest.get("evaluationCriteria"), H1_EVALUATION_CRITERIA,
        f"{manifest_path}.evaluationCriteria")

    plan = manifest.get("qualificationPlan")
    require_exact_keys(
        plan, {"evaluationSeeds", "evaluationEnvironments",
               "episodesPerReport"}, f"{manifest_path}.qualificationPlan")
    require(plan["evaluationSeeds"] == H1_EVALUATION_SEEDS
            and plan["evaluationEnvironments"] == H1_EVALUATION_ENVIRONMENTS
            and plan["episodesPerReport"] == H1_EPISODES_PER_REPORT,
            f"{manifest_path}: locked 4 x 512 qualification plan changed")

    metadata = context.metadata
    state = context.training_state
    require(manifest.get("observationDimension")
            == metadata.get("observationDimension") == 69
            and manifest.get("actionDimension")
            == metadata.get("actionDimension") == 19,
            f"{manifest_path}: H1 policy dimensions changed")
    require(f32_bits(manifest.get("simulationStepSeconds"))
            == f32_bits(metadata.get("simulationStep")) == f32_bits(0.005),
            f"{manifest_path}: H1 simulation step changed")
    require(manifest.get("controlDecimation")
            == metadata.get("controlDecimation") == 4
            and manifest.get("maxEpisodeSteps")
            == metadata.get("maxEpisodeSteps") == 1000
            and manifest.get("inferenceBatchSize")
            == metadata.get("inferenceBatchSize")
            == H1_EVALUATION_ENVIRONMENTS,
            f"{manifest_path}: H1 rollout contract changed")
    require(type(manifest.get("targetTrainingUpdates")) is int
            and manifest["targetTrainingUpdates"] == 0
            and type(manifest.get("targetTrainingEnvironmentSteps")) is int
            and manifest["targetTrainingEnvironmentSteps"] == 0,
            f"{manifest_path}: candidate is not optimizer-free")
    assert_json_equal(state, {
        "completedUpdates": 0,
        "environmentSteps": 0,
        "optimizerSteps": 0,
    }, f"{manifest_path}: candidate training state")

    parent = repository_path(
        root, H1_PARENT_DIRECTORY,
        f"{context.entry.selection_id} parent checkpoint")
    require(parent.is_dir(), f"{manifest_path}: parent checkpoint is missing")
    for directory, names, label in (
        (parent, ("metadata.json", "policy.safetensors", "training-state.json"),
         "parent checkpoint"),
        (context.directory,
         ("metadata.json", "policy.safetensors", "training-state.json",
          "deployment-manifest.json", "requalification-manifest.json"),
         "candidate checkpoint"),
    ):
        for name in names:
            require(not (directory / name).is_symlink(),
                    f"{label} file must not be a symlink: {name}")

    parent_metadata = load_json(parent / "metadata.json")
    parent_state = load_json(parent / "training-state.json")
    require(parent_metadata.get("task") == H1_TASK
            and parent_metadata.get("taskRevision") == H1_SOURCE_REVISION,
            f"{manifest_path}: parent task identity changed")
    require(config_equal(parent_metadata.get("taskConfiguration"),
                         H1_TASK_CONFIGURATION),
            f"{manifest_path}: parent task configuration changed")
    require(type(parent_state.get("completedUpdates")) is int
            and type(parent_state.get("environmentSteps")) is int
            and type(manifest.get("parentTrainingUpdates")) is int
            and manifest["parentTrainingUpdates"]
                == parent_state["completedUpdates"]
            and type(manifest.get("parentTrainingEnvironmentSteps")) is int
            and manifest["parentTrainingEnvironmentSteps"]
                == parent_state["environmentSteps"],
            f"{manifest_path}: parent training counters changed")

    hashes = {
        "parentPolicySHA256": parent / "policy.safetensors",
        "candidatePolicySHA256": context.directory / "policy.safetensors",
        "parentMetadataSHA256": parent / "metadata.json",
        "candidateMetadataSHA256": context.directory / "metadata.json",
        "parentTrainingStateSHA256": parent / "training-state.json",
        "candidateTrainingStateSHA256": context.directory / "training-state.json",
    }
    for field, path in hashes.items():
        expected_hash = require_sha256(
            manifest.get(field), f"{manifest_path}.{field}")
        require(sha256_file(path) == expected_hash,
                f"{manifest_path}: {field} does not match {path}")
    require((parent / "policy.safetensors").read_bytes()
            == (context.directory / "policy.safetensors").read_bytes()
            and manifest["parentPolicySHA256"]
            == manifest["candidatePolicySHA256"] == H1_POLICY_SHA256,
            f"{manifest_path}: policy lineage or pinned release digest changed")

    parent_fingerprint = checkpoint_fingerprint(parent)
    require(require_sha256(
        manifest.get("parentCheckpointFingerprint"),
        f"{manifest_path}.parentCheckpointFingerprint")
            == parent_fingerprint == H1_PARENT_FINGERPRINT,
        f"{manifest_path}: parent checkpoint fingerprint mismatch")
    require(require_sha256(
        manifest.get("candidateCheckpointFingerprint"),
        f"{manifest_path}.candidateCheckpointFingerprint")
            == context.fingerprint == H1_CANDIDATE_FINGERPRINT,
        f"{manifest_path}: candidate checkpoint fingerprint mismatch")
    require(parent_fingerprint != context.fingerprint,
            f"{manifest_path}: revision transfer did not change checkpoint identity")

    # Reconstruct the only allowed target-side metadata/state from the parent.
    expected_metadata = deepcopy(parent_metadata)
    expected_metadata["taskRevision"] = H1_TARGET_REVISION
    expected_metadata["taskConfiguration"] = deepcopy(H1_TASK_CONFIGURATION)
    expected_metadata["inferenceBatchSize"] = H1_EVALUATION_ENVIRONMENTS
    expected_ppo = expected_metadata.get("ppo")
    require(isinstance(expected_ppo, dict),
            f"{manifest_path}: parent PPO metadata is missing")
    expected_ppo["initializationCheckpoint"] = H1_PARENT_DIRECTORY
    expected_ppo["updates"] = 0
    assert_json_equal(metadata, expected_metadata,
                      f"{manifest_path}: candidate metadata transform")
    require(manifest["sourceTaskRevision"] == parent_metadata["taskRevision"]
            and manifest["targetTaskRevision"] == metadata["taskRevision"],
            f"{manifest_path}: manifest revisions do not bind checkpoint bytes")

    qualification = manifest.get("qualification")
    require_exact_keys(
        qualification, {"reports", "aggregate"},
        f"{manifest_path}.qualification")
    report_evidence = qualification.get("reports")
    require(isinstance(report_evidence, list)
            and len(report_evidence) == len(H1_EVALUATION_SEEDS),
            f"{manifest_path}: sealed report count changed")
    expected_report_names = {
        f"eval-seed-{seed}.json" for seed in H1_EVALUATION_SEEDS
    }
    qualification_directory = context.directory / "qualification"
    require(qualification_directory.is_dir()
            and not qualification_directory.is_symlink(),
            f"{manifest_path}: qualification directory is invalid")
    require({path.name for path in qualification_directory.iterdir()}
            == expected_report_names | {"aggregate.json"},
            f"{manifest_path}: qualification directory has missing or extra files")

    for item, seed in zip(report_evidence, H1_EVALUATION_SEEDS):
        label = f"{manifest_path}.qualification.report[{seed}]"
        require_exact_keys(item, {"evaluationSeed", "file", "sha256"}, label)
        expected_file = f"qualification/eval-seed-{seed}.json"
        require(item.get("evaluationSeed") == seed
                and item.get("file") == expected_file,
                f"{label}: seed or canonical file name changed")
        report_path = sealed_bundle_file(
            context.directory, item["file"], label)
        digest = require_sha256(item.get("sha256"), f"{label}.sha256")
        expected_successes, expected_digest = H1_REPORT_RESULTS[seed]
        require(digest == expected_digest
                and sha256_file(report_path) == digest,
                f"{label}: sealed report SHA-256 mismatch")
        report = load_json(report_path)
        require(report.get("evaluationSeed") == seed
                and report.get("evaluationEnvironments")
                    == H1_EVALUATION_ENVIRONMENTS
                and report.get("episodes") == H1_EPISODES_PER_REPORT
                and report.get("checkpointDirectory") == H1_CANDIDATE_DIRECTORY
                and type(report.get("trainingUpdates")) is int
                and report["trainingUpdates"] == 0
                and type(report.get("trainingEnvironmentSteps")) is int
                and report["trainingEnvironmentSteps"] == 0
                and type(report.get("successes")) is int
                and report["successes"] == expected_successes,
                f"{label}: report differs from the locked plan or zero-update lineage")
        mean_length = report.get("meanEpisodeLength")
        require(isinstance(mean_length, (int, float))
                and not isinstance(mean_length, bool)
                and 0 <= mean_length <= metadata["maxEpisodeSteps"],
                f"{label}: mean episode length is outside the task horizon")

    aggregate_evidence = qualification.get("aggregate")
    aggregate_label = f"{manifest_path}.qualification.aggregate"
    require_exact_keys(
        aggregate_evidence, {"file", "sha256"}, aggregate_label)
    require(aggregate_evidence.get("file") == "qualification/aggregate.json",
            f"{aggregate_label}: canonical aggregate path changed")
    aggregate_path = sealed_bundle_file(
        context.directory, aggregate_evidence["file"], aggregate_label)
    aggregate_hash = require_sha256(
        aggregate_evidence.get("sha256"), f"{aggregate_label}.sha256")
    require(aggregate_hash == H1_AGGREGATE_SHA256
            and sha256_file(aggregate_path) == aggregate_hash,
            f"{aggregate_label}: sealed aggregate SHA-256 mismatch")
    aggregate = load_json(aggregate_path)
    require(aggregate.get("evaluationSeeds") == H1_EVALUATION_SEEDS
            and aggregate.get("evaluationEnvironments")
                == H1_EVALUATION_ENVIRONMENTS
            and aggregate.get("requiredRuns") == 4
            and aggregate.get("runs") == 4
            and aggregate.get("requiredEpisodesPerRun")
                == H1_EPISODES_PER_REPORT
            and aggregate.get("totalEpisodes")
                == 4 * H1_EPISODES_PER_REPORT
            and type(aggregate.get("totalSuccesses")) is int
            and aggregate["totalSuccesses"] == H1_TOTAL_SUCCESSES
            and aggregate.get("acceptedRuns") == 4
            and aggregate.get("checkpointDirectory") == H1_CANDIDATE_DIRECTORY
            and aggregate.get("initializationCheckpoint")
                == H1_PARENT_DIRECTORY,
            f"{aggregate_path}: aggregate differs from the locked H1 plan")

    verify_deployment_manifest(context, required=True, exact=True)
    return aggregate_path


def verify(root: Path) -> tuple[int, int, int]:
    root = root.resolve()
    accepted = catalog_entries(root)
    aggregate_paths: set[Path] = set()
    report_count = 0
    for entry in accepted:
        context = verify_checkpoint(root, entry)
        evidence_path = repository_path(
            root, entry.evidence_relative_path,
            f"{entry.selection_id} evidence",
        )
        require(evidence_path.is_file(),
                f"{entry.selection_id}: missing evidence {evidence_path}")
        evidence = load_json(evidence_path)
        if entry.selection_id == H1_REQUALIFIED_SELECTION:
            aggregate_path = verify_h1_requalification(
                root, context, evidence_path)
            aggregate_paths.add(aggregate_path.resolve())
            report_count += verify_aggregate(aggregate_path, context)
            continue
        if evidence.get("scope") == "single_checkpoint_across_evaluation_seeds":
            siblings = sorted(evidence_path.parent.glob("*aggregate*.json"))
            require(evidence_path in siblings,
                    f"{entry.selection_id}: aggregate evidence naming is not canonical")
            for aggregate_path in siblings:
                aggregate_paths.add(aggregate_path.resolve())
                report_count += verify_aggregate(aggregate_path, context)
        elif EVALUATION_KEYS.issubset(evidence):
            require_evaluation_report(
                evidence, context, str(evidence_path), minimum_episodes=512
            )
            report_count += 1
        else:
            raise VerificationError(
                f"{entry.selection_id}: unsupported accepted evidence shape "
                f"at {evidence_path}"
            )
    return len(accepted), len(aggregate_paths), report_count


def main() -> int:
    root = DEFAULT_ROOT
    if len(sys.argv) == 3 and sys.argv[1] == "--root":
        root = Path(sys.argv[2])
    elif len(sys.argv) != 1:
        print(f"usage: {Path(sys.argv[0]).name} [--root repository]", file=sys.stderr)
        return 2
    try:
        accepted, aggregates, reports = verify(root)
    except (VerificationError, KeyError, OverflowError, struct.error,
            TypeError, ValueError) as error:
        print(f"policy evidence verification failed: {error}", file=sys.stderr)
        return 1
    print(
        "verified policy evidence: "
        f"{accepted} accepted catalog entries, {aggregates} robustness aggregates, "
        f"{reports} evaluation reports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
