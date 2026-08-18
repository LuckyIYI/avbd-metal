#!/usr/bin/env python3
"""Hermetically verify every accepted Policy Replay result.

This check deliberately does not import MLX or execute a policy. It verifies the
immutable boundary around an already evaluated policy instead:

* accepted and external-parity entries are discovered from
  ``PolicyReplayCatalog.swift``;
* tracked checkpoint identity is recomputed from the exact replay-semantic files;
* evaluation reports must match checkpoint task/revision/configuration/lineage;
* checkpoint robustness aggregates are rebuilt from their raw reports with the
  same Float32 quantile and pooling semantics as ``VectorPPO.swift``; and
* deployment-manifest hashes are checked when a checkpoint exposes them; and
* an optional built-app check rejects missing, historical, development, or
  otherwise unallowlisted checkpoint payloads.

An accepted evidence shape that this verifier does not understand fails closed,
so extending the release catalog requires extending this contract as well.
"""

from __future__ import annotations

import argparse
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

UNITREE_H1_SELECTION = "unitree-h1-sim2sim-v0"
UNITREE_H1_MANIFEST_SHA256 = (
    "9f434828cf2b2ede587bced686a22d30c3df6b048e631e94641bafeb7a45d117"
)
UNITREE_H1_WEIGHTS_SHA256 = (
    "cb51db3e4ccbecc0d9a863173640f8cb8b5a5fb821bc1db9024c7957297ff4ee"
)
UNITREE_H1_LICENSE_SHA256 = (
    "98335465f43a20b5850e4651db6e74c4aa1e9fc8e8813d38f345178045c0da50"
)
UNITREE_H1_SOURCE_REVISION = "276801e46c5d433564f24658bac64f254b7d2d4b"
UNITREE_H1_SOURCE_CHECKPOINT_SHA256 = (
    "44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb"
)

H1_REQUALIFIED_SELECTION = "humanoid-isaac-flat-v2"
H1_TASK = "humanoid-isaac-flat-v0"
H1_SOURCE_REVISION = 1_000_011
H1_TARGET_REVISION = 2_000_011
H1_PARENT_DIRECTORY = "checkpoints/humanoid-isaac-flat-v1"
H1_CANDIDATE_DIRECTORY = (
    "runs/humanoid-isaac-flat-v2/requalification-r2000011/candidate"
)
H1_DECLARED_SOURCE_COMMIT = "c5cc074163bb33bdc84dd7fb5390afba0156937c"
H1_PARENT_FINGERPRINT = (
    "85571805cc7b688970cf5497beb5916be8fb3b1fcb7855207af6f55b208c7fd2"
)
H1_CANDIDATE_FINGERPRINT = (
    "00bc782d1845ddde94282b46f0d7fa2732feeb4a8e52215a5abe62128bccc756"
)
H1_POLICY_SHA256 = (
    "3e0a21600afd6ee2e50383ed33f69007f9855c3f8b85ec0b52c4f2acc2c285ae"
)
H1_METADATA_SHA256 = (
    "16a24679f2f373bfc8b8285e15d2e9c40a9bb41cf5e1f42413d05ac5b63d2f85"
)
H1_TRAINING_STATE_SHA256 = (
    "15857f493c4a481825b008864879d53be5d398865a33c2249adfbbabe6d51e3e"
)
H1_DEPLOYMENT_MANIFEST_SHA256 = (
    "cb04233bd11bcc8dc3e0d2e1f0d6cc2e1ec27d4318344c7ea021d8b117be5d59"
)
H1_REQUALIFICATION_MANIFEST_SHA256 = (
    "c05d59a4592b7bdc896dca1c90db47a149b84352f593281f1b74b022e07031e9"
)
H1_AGGREGATE_SHA256 = (
    "b1b5975c59147878358fd90e224180172a3df890b2306f352ab37c84d43e2c8c"
)
H1_REPORT_RESULTS = {
    51_001: (506,
             "e3cf0b147de1b4b1c0d07db35944229b86fcb9d11cbaaee39c49dbe5b11dae02"),
    51_002: (507,
             "e326c95f2c39f72dd3273944f40b08ee6d7299d76550c18a9e3f62d2b4108142"),
    51_003: (505,
             "9901a6f52259f0735fa4952d1c3155cf0bf7452d4829b16d767a51a1870a2293"),
    51_004: (510,
             "1c60000c8332805c43d897ad796df1333eda5abf9038e5a16e8f88cd79d16083"),
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
H1_HISTORICAL_V0_FINGERPRINT = (
    "d6b5d416e7f7d75fa2b9b9dd33f78ae387e3f2a8139aa6d25a69e5dbcae777ab"
)
H1_HISTORICAL_V0_FILES = {
    "metadata.json":
        "30bceef674cf25d99bb8f1e0e3ce9867843ff926755ecfd27383915acde9a1ce",
    "policy.safetensors": H1_POLICY_SHA256,
    "training-state.json":
        "56bb45619a02264a02199d0c0ac0d96ff7fb9b3c9c8c6c58fc55756881c262e3",
    "evaluation.json":
        "c9652feee9b199e92156158bd891c5672c0deb29e97e9b3a292ff0c6bbd520bc",
}
H1_HISTORICAL_V1_FILES = {
    "metadata.json":
        "b0058de2a89ef4e27520d2bc1c00a9822c01bf8d8ac5a03a1bce31596e4d7be4",
    "policy.safetensors": H1_POLICY_SHA256,
    "training-state.json": H1_TRAINING_STATE_SHA256,
    "deployment-manifest.json":
        "9ec98be0db3149238693327eebc6eeab3c55536f3b7ea34d68521031674608fa",
    "requalification-manifest.json":
        "01628ae5d07d75a4538cbcf085b27314b2e72e4c5324f6bd4249d6297e03da03",
    "qualification/aggregate.json":
        "a3e54308ac978e68509ff1ea437a7f908be3b2c2a0df6699755e03a98d7e3f6d",
    "qualification/eval-seed-51001.json":
        "2c48696385668a22417bfb2d811adb39c05e6a501ac3ef7bf74bef2d046d3815",
    "qualification/eval-seed-51002.json":
        "e0c4c03dcbfa18b6258f87ed135bf70304e97bc4640a6d4858026513ea57aba9",
    "qualification/eval-seed-51003.json":
        "7412147895ddb1a48eec036a7e9d79bdd2379139b4997e62a6abbddf5e9fb124",
    "qualification/eval-seed-51004.json":
        "015bbb14ec3c9d1c7462ed1cd584f8af84685a02f0818f43428df6d6559b9e60",
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
REQUALIFICATION_V2_MANIFEST_KEYS = (
    REQUALIFICATION_MANIFEST_KEYS | {"qualificationMatrix"}
)
ARACHNE_EPOCH2_REVISION = 2_000_006
ARACHNE_QUALIFICATION_PROFILE = {
    "arachne15-velocity-v0": {
        "selectionID": "arachne15-velocity-v1",
        "checkpointRelativeDirectory": "arachne15-velocity-v1",
        "evidenceRelativePath": (
            "checkpoints/arachne15-velocity-v1/"
            "requalification-manifest.json"
        ),
        "parentDirectory": "checkpoints/arachne15-velocity-v0",
        "candidateDirectory": (
            "runs/arachne15-velocity-v1/"
            "requalification-r2000006/candidate"
        ),
        "declaredSourceCommit": (
            "a941cfe37c0f285105a192cc48192c85909a2d59"
        ),
        "parentFingerprint":
            "aed643b062df4e0e07e70998212720909bc1b25229455489ba28d4319d202524",
        "candidateFingerprint":
            "97f79641c8b7acf87c903b9d6baf739a5dc3c2536e52cb0e44121260133d79d5",
        "parentPolicySHA256":
            "a41b162b2bb922605e29a487a736f98b31416457807fc30a30c4e52014bf0638",
        "parentFiles": {
            "evaluation.json":
                "25b3dc73a57ac73711515ca648a82f20b8ce32cd1c39d8bd7f6b6659510fac26",
            "metadata.json":
                "1af26ab822bae72db6ed64d4f19dfe078b2cd333a010ee4488f4c22fbe8e307d",
            "policy.safetensors":
                "a41b162b2bb922605e29a487a736f98b31416457807fc30a30c4e52014bf0638",
            "training-state.json":
                "2dd66c4d70bd34a72050b3c938b74f7361066c1ac1a935bfe797c68add00131a",
        },
        "candidateFiles": {
            "deployment-manifest.json":
                "7295cf74dc9576a8b2bce74eeae0beb2932af96c5ff0617b0b99b82796b5039c",
            "metadata.json":
                "97e3d9325147fa620a8b709e2bbd51a33be34dacbb400ec21691fdfe050d52da",
            "policy.safetensors":
                "a41b162b2bb922605e29a487a736f98b31416457807fc30a30c4e52014bf0638",
            "requalification-manifest.json":
                "ba2b8dbe2d34e2c27a3a074dcdea91decc1bc57e08855293a9fbd149a227f159",
            "training-state.json":
                "15857f493c4a481825b008864879d53be5d398865a33c2249adfbbabe6d51e3e",
        },
        "suites": {
            "nominal": {
                "seeds": [61_001, 61_002, 61_003, 61_004],
                "successes": [512, 512, 512, 512],
                "totalSuccesses": 2_048,
                "aggregateSHA256":
                    "5a2383122b8c1ac45cf62b2a5010bf28de13e4d38f600d4c8766ed201754647c",
                "reportSHA256": [
                    "a425b6304acfd329a3764a8c3f83293f952cb00e4a03b2f5884e94cd18fa0943",
                    "6f15108cf351f02607fd8425fefc80864d026bed08d177396dff0d5330160865",
                    "36fee36f871e8f0422914c89411aa8ebec6bf6d4818420f771a56d359b2078fd",
                    "fc32a6c51028ba3fe938fee44a8c425725e608c98eedc538b7c2f4c211e21ac0",
                ],
            },
            "validation-collision": {
                "seeds": [61_501, 61_502, 61_503, 61_504],
                "successes": [512, 512, 512, 512],
                "totalSuccesses": 2_048,
                "aggregateSHA256":
                    "6ecf2f14837705ce85f6473341c6a3ba0cc09f2ed903a2b5bcdaf2c9d315fd99",
                "reportSHA256": [
                    "b24af2be1f97f7c2d3e9e4c57c80fb60ecd92e8ab5015d7c006385be85e5ac8d",
                    "72d802a719f69f257e325c51f3d2c4c9169e4c08ef96b8f76ec71efcdada67c9",
                    "f626f25b0963ac022c11ed3242c63155b7254b4be574ff335318932ce1b0ae25",
                    "a29ddc1347a6fcf8b7c6fcf570e004d1e0d815008bf4c06c5ea51d7b3692d82d",
                ],
            },
        },
    },
    "arachne15-goal-v0": {
        "selectionID": "arachne15-goal-v1",
        "checkpointRelativeDirectory": "arachne15-goal-v1",
        "evidenceRelativePath": (
            "checkpoints/arachne15-goal-v1/"
            "requalification-manifest.json"
        ),
        "parentDirectory": "checkpoints/arachne15-goal-v0",
        "candidateDirectory": (
            "runs/arachne15-goal-v1/"
            "requalification-r2000006/candidate"
        ),
        "declaredSourceCommit": (
            "a941cfe37c0f285105a192cc48192c85909a2d59"
        ),
        "parentFingerprint":
            "30c125b7f01b73bdd1524bc96cf8deb5e8a09897593a49e87aa6ce96f16d3027",
        "candidateFingerprint":
            "923e07c286f4fdb186b30a6fd95469e6848f4fec4ca1e3811320424b94c9dc02",
        "parentPolicySHA256":
            "9521c03cab6fc9e829cd2664fa0e086f69720d4aa46b1e5b893776a4df072c14",
        "parentFiles": {
            "deployment-manifest.json":
                "47a8c7a20f7cc405717693f16f9830a27a78e308545abdb0c5870eaeb57295ff",
            "metadata.json":
                "01e3f6d3983d522f82e951450a94ff5a4d5c3a9a31584793d64510b9e254dd32",
            "policy.safetensors":
                "9521c03cab6fc9e829cd2664fa0e086f69720d4aa46b1e5b893776a4df072c14",
            "training-state.json":
                "8286ef59443b9b00c110064b938e897dec556582de1bd017871329e2e47dc087",
        },
        "candidateFiles": {
            "deployment-manifest.json":
                "e7d747a41b3f724940bbe42d92dc38de8798dafbc7d39909e4ad1cf10ae1e127",
            "metadata.json":
                "8cf417d48ec00849c78feedbd3f1881c87b54e3496bff07e4be1883a829cffff",
            "policy.safetensors":
                "9521c03cab6fc9e829cd2664fa0e086f69720d4aa46b1e5b893776a4df072c14",
            "requalification-manifest.json":
                "1546b0ddaaaec8995888522723c0ef3f2f4a48ed3d72abecad5539777e66a192",
            "training-state.json":
                "15857f493c4a481825b008864879d53be5d398865a33c2249adfbbabe6d51e3e",
        },
        "suites": {
            "nominal": {
                "seeds": [62_001, 62_002, 62_003, 62_004],
                "successes": [487, 492, 485, 478],
                "totalSuccesses": 1_942,
                "aggregateSHA256":
                    "3285ec8bb356ab3a0db4676a46bc792e849d9e07c8e4ba9339115c347f872f16",
                "reportSHA256": [
                    "85a9ecf8e164ead072ecad6f07427b7b9867d1942324376a7a66e01917f042a5",
                    "6f39b8b26904dbfb1d74193385d597b633c1bc658684dd1cb7cf8881ebf87f12",
                    "47c1aa017ce8efd71c31c80660d1f7ef9262a32b9489e31532ac384c4a69ee2b",
                    "2e0296e20e93cbb114091f75b25abd69713f16314b5782f25c8cf8b0a17b606c",
                ],
            },
            "validation-collision": {
                "seeds": [63_001, 63_002, 63_003, 63_004],
                "successes": [493, 481, 473, 488],
                "totalSuccesses": 1_935,
                "aggregateSHA256":
                    "49cea9406763130f30d16964c531d8172d998442d8e3aa5b2f8982c93713c304",
                "reportSHA256": [
                    "cfa102e9161386309385bc580caf474d9797f59b994191d6432da9ae1952cb55",
                    "028d179c4cf5f020f0c479dd8d3df0db5ec4c45ba446f883c6ebb159d113b56b",
                    "3667c68d9c1cc46fb499f92ae2d73faded64f3425e3ebbf4b25f4ac221bc9af4",
                    "19982933065541eaacc7be8d88ce66420fae3d9c964520d6f91d2313c515ce89",
                ],
            },
        },
    },
}


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    selection_id: str
    task_id: str
    runtime: str
    qualification: str
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


def arachne_candidate_files(profile: dict[str, Any]) -> dict[str, str]:
    """Return the complete immutable file map for one Arachne v1 bundle."""
    expected = dict(profile["candidateFiles"])
    for suite_id, suite in profile["suites"].items():
        seeds = suite["seeds"]
        successes = suite["successes"]
        report_hashes = suite["reportSHA256"]
        require(len(seeds) == len(successes) == len(report_hashes) == 4,
                f"Arachne {suite_id} pinned outcome vector is incomplete")
        expected[f"qualification/{suite_id}/aggregate.json"] = (
            suite["aggregateSHA256"]
        )
        for seed, digest in zip(seeds, report_hashes):
            expected[f"qualification/{suite_id}/eval-seed-{seed}.json"] = digest
    return expected


def verify_pinned_file_tree(directory: Path, expected: dict[str, str],
                            label: str) -> None:
    """Bind an immutable release directory to exact paths and exact bytes."""
    require(directory.is_dir(), f"{label}: directory is missing")
    expected_files = {Path(relative) for relative in expected}
    require(len(expected_files) == len(expected),
            f"{label}: pinned file map contains duplicate paths")
    expected_directories = {Path(".")}
    for relative in expected_files:
        require(not relative.is_absolute()
                and all(part not in {"", ".", ".."}
                        for part in relative.parts),
                f"{label}: non-canonical pinned path {relative}")
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent)
            parent = parent.parent

    actual_files: set[Path] = set()
    actual_directories: set[Path] = {Path(".")}
    for path in directory.rglob("*"):
        require(not path.is_symlink(),
                f"{label}: pinned tree contains a symlink: {path}")
        relative = path.relative_to(directory)
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise VerificationError(
                f"{label}: pinned tree contains a special file: {path}")
    require(actual_files == expected_files,
            f"{label}: file inventory changed (missing="
            f"{sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)})")
    require(actual_directories == expected_directories,
            f"{label}: directory inventory changed (missing="
            f"{sorted(expected_directories - actual_directories)}, "
            f"extra={sorted(actual_directories - expected_directories)})")
    for relative, expected_digest in expected.items():
        require(SHA256_PATTERN.fullmatch(expected_digest) is not None,
                f"{label}: invalid pinned SHA-256 for {relative}")
        path = directory / relative
        require(sha256_file(path) == expected_digest,
                f"{label}: SHA-256 changed for {relative}")


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


def catalog_entries(root: Path) -> tuple[list[CatalogEntry], list[CatalogEntry]]:
    path = root / "Sources/AVBDLearn/PolicyReplayCatalog.swift"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    declaration = source.find("public static let entries")
    require(declaration >= 0, "PolicyReplayCatalog.entries is missing")
    assignment = source.find("=", declaration)
    closure_start = source.find("{", assignment)
    require(assignment >= 0 and closure_start >= 0,
            "PolicyReplayCatalog.entries is not an initialized closure")
    inner_declaration = re.search(
        r"\blet\s+entries\s*:\s*\[PolicyReplayCatalogEntry\]\s*=",
        source[closure_start:],
    )
    require(inner_declaration is not None,
            "PolicyReplayCatalog.entries closure has no literal entries array")
    inner_assignment = closure_start + inner_declaration.end() - 1
    array_start = source.find("[", inner_assignment + 1)
    require(array_start >= 0,
            "PolicyReplayCatalog.entries inner assignment is not an array")
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
    external_parity: list[CatalogEntry] = []
    all_selection_ids: set[str] = set()
    for block in blocks:
        selection_id = swift_string(
            argument_expression(block, "selectionID"), "selectionID")
        require(selection_id is not None and selection_id not in all_selection_ids,
                f"invalid or duplicate Policy Replay selection {selection_id!r}")
        all_selection_ids.add(selection_id)
        qualification = argument_expression(block, "qualification")
        if qualification not in {".accepted", ".externalParityVerified"}:
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
                f"qualified selection {selection_id} must name task, checkpoint and evidence")
        entry = CatalogEntry(
            selection_id=selection_id,
            task_id=task_id,
            runtime=runtime,
            qualification=qualification,
            checkpoint_relative_directory=checkpoint,
            evidence_relative_path=evidence,
        )
        if qualification == ".accepted":
            require(runtime == ".nativeMLX",
                    f"accepted selection {selection_id} uses unsupported runtime {runtime}")
            accepted.append(entry)
        else:
            require(runtime == ".unitreeRecurrentMLX",
                    f"external-parity selection {selection_id} uses unsupported runtime {runtime}")
            external_parity.append(entry)
    require(bool(accepted) or bool(external_parity),
            "PolicyReplayCatalog contains no verified policy evidence")
    return accepted, external_parity


def require_numeric_matrix(value: Any, rows: int, columns: int,
                           label: str) -> None:
    require(isinstance(value, list) and len(value) == rows,
            f"{label}: expected {rows} rows")
    for row_index, row in enumerate(value):
        require(isinstance(row, list) and len(row) == columns,
                f"{label}[{row_index}]: expected {columns} values")
        require(all(isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(float(item)) for item in row),
                f"{label}[{row_index}]: values must be finite numbers")


def verify_external_parity(root: Path, entry: CatalogEntry) -> None:
    require(entry.selection_id == UNITREE_H1_SELECTION,
            f"{entry.selection_id}: unsupported external parity contract")
    require(entry.task_id == UNITREE_H1_SELECTION,
            "Unitree external parity entry changed task identity")
    require(entry.checkpoint_relative_directory == "external/unitree-h1"
            and entry.evidence_relative_path
            == "checkpoints/external/unitree-h1/manifest.json",
            "Unitree external parity entry changed its pinned bundle")
    directory = repository_path(
        root, f"checkpoints/{entry.checkpoint_relative_directory}",
        f"{entry.selection_id} checkpoint",
    )
    require(directory.is_dir(),
            f"{entry.selection_id}: checkpoint directory is missing")
    evidence = repository_path(
        root, entry.evidence_relative_path, f"{entry.selection_id} evidence")
    require(evidence == (directory / "manifest.json").resolve(),
            f"{entry.selection_id}: parity evidence must be its manifest")
    require(sha256_file(evidence) == UNITREE_H1_MANIFEST_SHA256,
            f"{evidence}: SHA-256 does not match the pinned Unitree release")
    manifest = load_json(evidence)
    require_exact_keys(manifest, {
        "schemaVersion", "format", "source", "robot", "weightsFile",
        "weightsSHA256", "network", "control", "observationLayout",
        "goldenSequence",
    }, str(evidence))
    require(manifest.get("schemaVersion") == 2
            and manifest.get("format") == "avbd-unitree-h1-lstm-v1"
            and manifest.get("robot") == "unitree-h1",
            f"{evidence}: unsupported Unitree parity manifest")

    weights = sealed_bundle_file(
        directory, manifest.get("weightsFile"), "Unitree policy weights")
    weights_hash = require_sha256(
        manifest.get("weightsSHA256"), "Unitree weightsSHA256")
    require(weights_hash == UNITREE_H1_WEIGHTS_SHA256,
            "Unitree weightsSHA256 does not match the pinned release")
    require(sha256_file(weights) == weights_hash,
            f"{weights}: SHA-256 does not match the parity manifest")

    source = manifest.get("source")
    require_exact_keys(source, {
        "project", "revision", "url", "checkpointSHA256", "license",
        "licenseFile", "licenseSHA256",
    }, "Unitree source")
    revision = source.get("revision")
    require(revision == UNITREE_H1_SOURCE_REVISION,
            "Unitree source revision does not match the pinned release")
    checkpoint_hash = require_sha256(
        source.get("checkpointSHA256"), "Unitree source checkpointSHA256")
    require(checkpoint_hash == UNITREE_H1_SOURCE_CHECKPOINT_SHA256,
            "Unitree source checkpointSHA256 does not match the pinned release")
    license_file = sealed_bundle_file(
        directory, source.get("licenseFile"), "Unitree source license")
    license_hash = require_sha256(
        source.get("licenseSHA256"), "Unitree source licenseSHA256")
    require(license_hash == UNITREE_H1_LICENSE_SHA256,
            "Unitree licenseSHA256 does not match the pinned release")
    require(sha256_file(license_file) == license_hash,
            f"{license_file}: SHA-256 does not match the parity manifest")

    network = manifest.get("network")
    require(isinstance(network, dict), "Unitree network must be an object")
    observation_dimension = network.get("observationDimension")
    action_dimension = network.get("actionDimension")
    hidden_dimension = network.get("hiddenDimension")
    require(type(observation_dimension) is int and observation_dimension > 0
            and type(action_dimension) is int and action_dimension > 0
            and type(hidden_dimension) is int and hidden_dimension > 0,
            "Unitree network dimensions must be positive integers")
    golden = manifest.get("goldenSequence")
    require_exact_keys(golden, {
        "absoluteTolerance", "inputs", "actions", "hiddenStates", "cellStates",
    }, "Unitree golden sequence")
    tolerance = golden.get("absoluteTolerance")
    require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
            and math.isfinite(float(tolerance)) and 0 < tolerance <= 1e-3,
            "Unitree golden tolerance is invalid")
    rows = len(golden.get("inputs")) if isinstance(golden.get("inputs"), list) else 0
    require(rows > 0, "Unitree golden sequence must not be empty")
    require_numeric_matrix(golden.get("inputs"), rows, observation_dimension,
                           "Unitree golden inputs")
    require_numeric_matrix(golden.get("actions"), rows, action_dimension,
                           "Unitree golden actions")
    require_numeric_matrix(golden.get("hiddenStates"), rows, hidden_dimension,
                           "Unitree golden hidden states")
    require_numeric_matrix(golden.get("cellStates"), rows, hidden_dimension,
                           "Unitree golden cell states")


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
        # Match PPOConfiguration.resolvedActionDistribution: historical
        # checkpoints decode a missing field as the squashed Gaussian policy.
        distribution = ppo.get("actionDistribution", "squashed-gaussian")
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
                "accepted H1 evidence must use the current requalified selection")
        require(config_equal(configuration, H1_TASK_CONFIGURATION),
                "accepted H1 evidence changed the task configuration")
        if f32(report["successRate"]) < f32(0.80):
            failures.append("success rate below threshold")
        minimum("episode/survived", 0.90)
        maximum("episode/linear_velocity_rmse_mps", 0.35)
        maximum("episode/yaw_rate_rmse_rps", 0.50)
        minimum_length_fraction = f32(0.90)
    else:
        require(revision in {6, ARACHNE_EPOCH2_REVISION}
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
        if f32(report["successRate"]) < f32(0.90):
            failures.append("success rate below threshold")
        minimum("episode/survived", 0.95)
        minimum("episode/minimum_foot_collider_clearance_m", -0.003)
        maximum("episode/linear_velocity_rmse_mps", 0.15)
        maximum("episode/yaw_rate_rmse_rps", 0.40)
        maximum("episode/foot_collider_penetration_rmse_m", 0.0005)
        maximum("episode/foot_collider_penetration_over_1mm_fraction", 0.025)
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


def verify_pinned_historical_h1_lineage(root: Path) -> None:
    """Keep the immutable v0/v1 source chain bound while v2 is accepted."""
    v0 = repository_path(
        root, "checkpoints/humanoid-isaac-flat-v0", "historical H1 v0")
    v1 = repository_path(
        root, H1_PARENT_DIRECTORY, "historical H1 v1")
    for directory, expected_files, label in (
        (v0, H1_HISTORICAL_V0_FILES, "historical H1 v0"),
        (v1, H1_HISTORICAL_V1_FILES, "historical H1 v1"),
    ):
        require(directory.is_dir(), f"{label}: checkpoint directory is missing")
        paths = list(directory.rglob("*"))
        require(all(not path.is_symlink() for path in paths),
                f"{label}: bundle must not contain symlinks")
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in paths if path.is_file()
        }
        require(actual_files == set(expected_files),
                f"{label}: bundle has missing or extra files")
        for relative, expected_hash in expected_files.items():
            require(sha256_file(directory / relative) == expected_hash,
                    f"{label}: pinned bytes changed for {relative}")

    require(checkpoint_fingerprint(v0) == H1_HISTORICAL_V0_FINGERPRINT,
            "historical H1 v0 checkpoint fingerprint changed")
    require(checkpoint_fingerprint(v1) == H1_PARENT_FINGERPRINT,
            "historical H1 v1 checkpoint fingerprint changed")
    v0_metadata = load_json(v0 / "metadata.json")
    v1_metadata = load_json(v1 / "metadata.json")
    require(v0_metadata.get("task") == H1_TASK
            and v0_metadata.get("taskRevision") == 1_000_010,
            "historical H1 v0 task identity changed")
    require(v1_metadata.get("task") == H1_TASK
            and v1_metadata.get("taskRevision") == H1_SOURCE_REVISION,
            "historical H1 v1 task identity changed")

    manifest = load_json(v1 / "requalification-manifest.json")
    require(manifest.get("sourceTaskRevision") == 1_000_010
            and manifest.get("targetTaskRevision") == H1_SOURCE_REVISION
            and manifest.get("parentCheckpointDirectory")
                == "checkpoints/humanoid-isaac-flat-v0"
            and manifest.get("parentCheckpointFingerprint")
                == H1_HISTORICAL_V0_FINGERPRINT
            and manifest.get("candidateCheckpointFingerprint")
                == H1_PARENT_FINGERPRINT,
            "historical H1 v0-to-v1 lineage changed")


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
            == (f"checkpoints/{H1_REQUALIFIED_SELECTION}/"
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
    verify_pinned_historical_h1_lineage(root)
    require(not manifest_path.is_symlink(),
            f"{manifest_path}: manifest must not be a symlink")
    pinned_release_files = {
        "metadata.json": H1_METADATA_SHA256,
        "policy.safetensors": H1_POLICY_SHA256,
        "training-state.json": H1_TRAINING_STATE_SHA256,
        "deployment-manifest.json": H1_DEPLOYMENT_MANIFEST_SHA256,
        "requalification-manifest.json":
            H1_REQUALIFICATION_MANIFEST_SHA256,
    }
    for relative, expected_hash in pinned_release_files.items():
        path = sealed_bundle_file(
            context.directory, relative,
            f"{context.entry.selection_id} pinned release file")
        require(sha256_file(path) == expected_hash,
                f"{path}: pinned release bytes changed")
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


def configuration_changed_fields(first: dict[str, Any],
                                 second: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key in sorted(first.keys() | second.keys()):
        if key not in first or key not in second:
            changed.append(key)
        elif f32_bits(first[key]) != f32_bits(second[key]):
            changed.append(key)
    return changed


def arachne_evaluation_criteria(task: str,
                                configuration: dict[str, Any]) -> dict[str, Any]:
    minimum_metrics: dict[str, float] = {
        "episode/survived": 0.95,
        "episode/minimum_foot_collider_clearance_m": -0.003,
    }
    maximum_metrics: dict[str, float] = {
        "episode/yaw_rate_rmse_rps": 0.40,
        "episode/foot_collider_penetration_rmse_m": 0.0005,
        "episode/foot_collider_penetration_over_1mm_fraction": 0.025,
    }
    if task == "arachne15-velocity-v0":
        maximum_metrics["episode/linear_velocity_rmse_mps"] = 0.15
        return {
            "minimumSuccessRate": 0.90,
            "minimumMeanEpisodeLengthFraction": 0.90,
            "minimumTaskMetrics": minimum_metrics,
            "maximumTaskMetrics": maximum_metrics,
        }
    require(task == "arachne15-goal-v0",
            f"unsupported schema-v2 task {task}")
    minimum_metrics.update({
        "episode/goal_reached": 0.90,
        "episode/goal_front_success_rate": 0.85,
        "episode/goal_near_success_rate": 0.85,
        "episode/goal_far_success_rate": 0.85,
    })
    direction = f32(configuration["maximumGoalDirectionAngle"])
    if direction > f32(f32(math.pi) / f32(4)):
        minimum_metrics["episode/goal_left_success_rate"] = 0.85
        minimum_metrics["episode/goal_right_success_rate"] = 0.85
    if direction > f32(f32(3) * f32(math.pi) / f32(4)):
        minimum_metrics["episode/goal_rear_success_rate"] = 0.85
    radius = f32(configuration["goalRadius"])
    maximum_metrics.update({
        "episode/final_goal_distance_m": f32(radius * f32(1.25)),
        "episode/minimum_goal_distance_m": radius,
    })
    return {
        "minimumSuccessRate": 0.90,
        "minimumMeanEpisodeLengthFraction": 0.10,
        "minimumTaskMetrics": minimum_metrics,
        "maximumTaskMetrics": maximum_metrics,
    }


def verify_schema2_requalification(
    root: Path, context: CheckpointContext, manifest_path: Path
) -> tuple[list[Path], int]:
    """Verify a generic multi-suite bundle and the pinned Arachne profile."""
    bundle_manifest = context.directory / "requalification-manifest.json"
    require(not bundle_manifest.is_symlink()
            and manifest_path.resolve() == bundle_manifest.resolve(),
            f"{context.entry.selection_id}: evidence must be the in-bundle "
            "non-symlink requalification manifest")
    manifest = load_json(manifest_path)
    require_exact_keys(manifest, REQUALIFICATION_V2_MANIFEST_KEYS,
                       str(manifest_path))
    require(manifest.get("schemaVersion") == 2,
            f"{manifest_path}: unsupported multi-suite schema")
    task = context.entry.task_id
    profile = ARACHNE_QUALIFICATION_PROFILE.get(task)
    require(profile is not None,
            f"{context.entry.selection_id}: schema-v2 release profile is not pinned")
    expected_checkpoint = repository_path(
        root, f"checkpoints/{profile['checkpointRelativeDirectory']}",
        f"{context.entry.selection_id} pinned checkpoint",
    )
    require(context.entry.selection_id == profile["selectionID"]
            and context.entry.checkpoint_relative_directory
                == profile["checkpointRelativeDirectory"]
            and context.entry.evidence_relative_path
                == profile["evidenceRelativePath"]
            and context.directory == expected_checkpoint
            and not (root / "checkpoints"
                     / profile["checkpointRelativeDirectory"]).is_symlink(),
            f"{context.entry.selection_id}: published catalog identity changed")
    verify_pinned_file_tree(
        context.directory, arachne_candidate_files(profile),
        f"{context.entry.selection_id} candidate",
    )
    metadata = context.metadata
    state = context.training_state
    require(metadata.get("taskRevision") == ARACHNE_EPOCH2_REVISION
            and manifest.get("task") == task
            and manifest.get("targetTaskRevision") == ARACHNE_EPOCH2_REVISION,
            f"{manifest_path}: Arachne epoch-2 task identity changed")
    require(manifest.get("changedFields") == REQUALIFICATION_CHANGED_FIELDS,
            f"{manifest_path}: permitted zero-update transform changed")
    source_commit = manifest.get("declaredSourceCommit")
    require(source_commit == profile["declaredSourceCommit"],
            f"{manifest_path}: declared source commit changed")
    candidate_relative = manifest.get("candidateCheckpointDirectory")
    require(candidate_relative == profile["candidateDirectory"]
            and candidate_relative != manifest.get("parentCheckpointDirectory"),
            f"{manifest_path}: candidate checkpoint path changed")
    assert_json_equal(manifest.get("taskConfiguration"),
                      metadata.get("taskConfiguration"),
                      f"{manifest_path}.taskConfiguration")
    require(manifest.get("observationDimension")
            == metadata.get("observationDimension")
            and manifest.get("actionDimension")
            == metadata.get("actionDimension")
            and f32_bits(manifest.get("simulationStepSeconds"))
            == f32_bits(metadata.get("simulationStep"))
            and manifest.get("controlDecimation")
            == metadata.get("controlDecimation")
            and manifest.get("maxEpisodeSteps")
            == metadata.get("maxEpisodeSteps")
            and manifest.get("inferenceBatchSize") == 128,
            f"{manifest_path}: rollout contract changed")
    require(manifest.get("targetTrainingUpdates") == 0
            and manifest.get("targetTrainingEnvironmentSteps") == 0,
            f"{manifest_path}: candidate is not optimizer-free")
    assert_json_equal(state, {
        "completedUpdates": 0,
        "environmentSteps": 0,
        "optimizerSteps": 0,
    }, f"{manifest_path}: candidate training state")

    parent_relative = manifest.get("parentCheckpointDirectory")
    require(parent_relative == profile["parentDirectory"]
            and manifest.get("sourceTaskRevision") == 6,
            f"{manifest_path}: pinned epoch-1 parent lineage changed")
    parent = repository_path(root, parent_relative,
                             f"{context.entry.selection_id} parent checkpoint")
    require(parent.is_dir() and not (root / parent_relative).is_symlink(),
            f"{manifest_path}: parent checkpoint is missing or aliased")
    verify_pinned_file_tree(
        parent, profile["parentFiles"],
        f"{context.entry.selection_id} epoch-1 parent",
    )
    parent_metadata = load_json(parent / "metadata.json")
    parent_state = load_json(parent / "training-state.json")
    require(parent_metadata.get("task") == task
            and parent_metadata.get("taskRevision")
                == manifest.get("sourceTaskRevision")
            and parent_metadata.get("taskRevision") != ARACHNE_EPOCH2_REVISION,
            f"{manifest_path}: parent task lineage changed")
    require(config_equal(parent_metadata.get("taskConfiguration"),
                         metadata.get("taskConfiguration")),
            f"{manifest_path}: zero-update transfer changed nominal config")
    require(manifest.get("parentTrainingUpdates")
            == parent_state.get("completedUpdates")
            and manifest.get("parentTrainingEnvironmentSteps")
            == parent_state.get("environmentSteps"),
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
        digest = require_sha256(manifest.get(field),
                                f"{manifest_path}.{field}")
        require(not path.is_symlink() and sha256_file(path) == digest,
                f"{manifest_path}: {field} does not bind {path}")
    require((parent / "policy.safetensors").read_bytes()
            == (context.directory / "policy.safetensors").read_bytes()
            and manifest["parentPolicySHA256"]
                == manifest["candidatePolicySHA256"]
                == profile["parentPolicySHA256"],
            f"{manifest_path}: policy bytes changed during requalification")
    require(checkpoint_fingerprint(parent)
            == manifest.get("parentCheckpointFingerprint")
            == profile["parentFingerprint"]
            and context.fingerprint
            == manifest.get("candidateCheckpointFingerprint")
            == profile["candidateFingerprint"],
            f"{manifest_path}: checkpoint fingerprint lineage changed")

    expected_metadata = deepcopy(parent_metadata)
    expected_metadata["taskRevision"] = ARACHNE_EPOCH2_REVISION
    expected_metadata["taskConfiguration"] = deepcopy(
        metadata["taskConfiguration"])
    expected_metadata["inferenceBatchSize"] = 128
    expected_ppo = expected_metadata.get("ppo")
    require(isinstance(expected_ppo, dict),
            f"{manifest_path}: parent PPO metadata is missing")
    expected_ppo["initializationCheckpoint"] = parent_relative
    expected_ppo["updates"] = 0
    assert_json_equal(metadata, expected_metadata,
                      f"{manifest_path}: candidate metadata transform")

    plan = manifest.get("qualificationPlan")
    require_exact_keys(plan, {"evaluationSeeds", "evaluationEnvironments",
                              "episodesPerReport"},
                       f"{manifest_path}.qualificationPlan")
    matrix = manifest.get("qualificationMatrix")
    require_exact_keys(matrix, {"suites", "comparisons"},
                       f"{manifest_path}.qualificationMatrix")
    suites = matrix.get("suites")
    comparisons = matrix.get("comparisons")
    require(isinstance(suites, list) and len(suites) == 2,
            f"{manifest_path}: Arachne requires nominal and collision suites")
    require(isinstance(comparisons, list) and len(comparisons) == 1,
            f"{manifest_path}: Arachne requires one cross-suite gate")
    suite_ids = ["nominal", "validation-collision"]
    require(list(profile["suites"]) == suite_ids,
            f"{manifest_path}: pinned suite order changed")
    expected_seeds = [profile["suites"][suite_id]["seeds"]
                      for suite_id in suite_ids]
    nominal_configuration = metadata.get("taskConfiguration")
    require(isinstance(nominal_configuration, dict)
            and f32_bits(nominal_configuration.get(
                "validationCollisionProfile")) == f32_bits(0),
            f"{manifest_path}: nominal collision profile must be zero")
    validation_configuration = deepcopy(nominal_configuration)
    validation_configuration["validationCollisionProfile"] = 1
    expected_configurations = [nominal_configuration, validation_configuration]
    expected_deltas = [[], ["validationCollisionProfile"]]
    for index, suite in enumerate(suites):
        label = f"{manifest_path}.qualificationMatrix.suites[{index}]"
        require_exact_keys(suite, {
            "id", "evaluationSeeds", "evaluationEnvironments",
            "episodesPerReport", "evaluationTaskConfiguration",
            "changedConfigurationFields", "evaluationCriteria",
        }, label)
        require(suite.get("id") == suite_ids[index]
                and suite.get("evaluationSeeds") == expected_seeds[index]
                and suite.get("evaluationEnvironments") == 128
                and suite.get("episodesPerReport") == 512,
                f"{label}: fixed suite plan changed")
        assert_json_equal(suite.get("evaluationTaskConfiguration"),
                          expected_configurations[index],
                          f"{label}.evaluationTaskConfiguration")
        require(suite.get("changedConfigurationFields")
                == expected_deltas[index]
                == configuration_changed_fields(
                    nominal_configuration, expected_configurations[index]),
                f"{label}: collision-profile delta changed")
        assert_json_equal(
            suite.get("evaluationCriteria"),
            arachne_evaluation_criteria(task, expected_configurations[index]),
            f"{label}.evaluationCriteria")
    require(plan == {
        "evaluationSeeds": expected_seeds[0],
        "evaluationEnvironments": 128,
        "episodesPerReport": 512,
    }, f"{manifest_path}: primary qualification alias changed")
    assert_json_equal(manifest.get("evaluationCriteria"),
                      suites[0].get("evaluationCriteria"),
                      f"{manifest_path}.evaluationCriteria")
    comparison = comparisons[0]
    require_exact_keys(comparison, {
        "baselineSuite", "evaluatedSuite", "maximumPooledSuccessRateDrop",
    }, f"{manifest_path}.qualificationMatrix.comparisons[0]")
    require(comparison.get("baselineSuite") == "nominal"
            and comparison.get("evaluatedSuite") == "validation-collision"
            and f32_bits(comparison.get("maximumPooledSuccessRateDrop"))
                == f32_bits(0.05),
            f"{manifest_path}: cross-suite gate changed")

    qualification = manifest.get("qualification")
    require_exact_keys(qualification, {"reports", "aggregate",
                                       "additionalSuites"},
                       f"{manifest_path}.qualification")
    additional = qualification.get("additionalSuites")
    require(isinstance(additional, list) and len(additional) == 1
            and additional[0].get("id") == "validation-collision",
            f"{manifest_path}: additional suite evidence changed")
    evidence_by_id = {
        "nominal": {
            "reports": qualification.get("reports"),
            "aggregate": qualification.get("aggregate"),
        },
        "validation-collision": additional[0],
    }
    qualification_directory = context.directory / "qualification"
    expected_files: set[Path] = set()
    aggregate_paths: list[Path] = []
    report_count = 0
    aggregates: dict[str, dict[str, Any]] = {}
    for index, suite_id in enumerate(suite_ids):
        suite = suites[index]
        pinned_suite = profile["suites"][suite_id]
        evidence = evidence_by_id[suite_id]
        require_exact_keys(evidence,
                           {"reports", "aggregate"}
                           | ({"id"} if suite_id != "nominal" else set()),
                           f"{manifest_path}.qualification.{suite_id}")
        report_evidence = evidence.get("reports")
        require(isinstance(report_evidence, list)
                and len(report_evidence) == 4,
                f"{manifest_path}: suite {suite_id} needs four reports")
        for report_index, (item, seed) in enumerate(zip(
                report_evidence, expected_seeds[index])):
            require_exact_keys(item, {"evaluationSeed", "file", "sha256"},
                               f"{manifest_path}.{suite_id}.report")
            relative = f"qualification/{suite_id}/eval-seed-{seed}.json"
            require(item.get("evaluationSeed") == seed
                    and item.get("file") == relative,
                    f"{manifest_path}: suite report path or seed changed")
            path = sealed_bundle_file(context.directory, relative,
                                      f"{manifest_path}.{suite_id}.report")
            require(sha256_file(path) == require_sha256(
                item.get("sha256"), f"{manifest_path}.{suite_id}.sha256"),
                f"{path}: sealed report digest mismatch")
            require(item.get("sha256")
                    == pinned_suite["reportSHA256"][report_index],
                    f"{path}: published report digest changed")
            report = load_json(path)
            require(report.get("evaluationSeed") == seed
                    and report.get("evaluationEnvironments") == 128
                    and report.get("episodes") == 512
                    and report.get("successes")
                        == pinned_suite["successes"][report_index]
                    and report.get("checkpointDirectory")
                        == manifest.get("candidateCheckpointDirectory")
                    and config_equal(report.get("evaluationTaskConfiguration"),
                                     expected_configurations[index])
                    and report.get("taskConfigurationTransferred")
                        == (index == 1),
                    f"{path}: report differs from its frozen suite")
            expected_files.add(Path(relative))
        aggregate_evidence = evidence.get("aggregate")
        require_exact_keys(aggregate_evidence, {"file", "sha256"},
                           f"{manifest_path}.{suite_id}.aggregate")
        aggregate_relative = f"qualification/{suite_id}/aggregate.json"
        require(aggregate_evidence.get("file") == aggregate_relative,
                f"{manifest_path}: aggregate path changed")
        require(aggregate_evidence.get("sha256")
                == pinned_suite["aggregateSHA256"],
                f"{manifest_path}: published {suite_id} aggregate changed")
        aggregate_path = sealed_bundle_file(
            context.directory, aggregate_relative,
            f"{manifest_path}.{suite_id}.aggregate")
        require(sha256_file(aggregate_path) == require_sha256(
            aggregate_evidence.get("sha256"),
            f"{manifest_path}.{suite_id}.aggregate.sha256"),
            f"{aggregate_path}: sealed aggregate digest mismatch")
        aggregate = load_json(aggregate_path)
        require(aggregate.get("evaluationSeeds") == expected_seeds[index]
                and aggregate.get("evaluationEnvironments") == 128
                and aggregate.get("requiredRuns") == 4
                and aggregate.get("runs") == 4
                and aggregate.get("requiredEpisodesPerRun") == 512
                and aggregate.get("totalEpisodes") == 2_048
                and aggregate.get("totalSuccesses")
                    == pinned_suite["totalSuccesses"]
                and aggregate.get("acceptedRuns") == 4
                and aggregate.get("allRunsPassed") is True
                and aggregate.get("robustAcrossEvaluationSeeds") is True
                and config_equal(aggregate.get("evaluationTaskConfiguration"),
                                 expected_configurations[index])
                and aggregate.get("taskConfigurationTransferred")
                    == (index == 1),
                f"{aggregate_path}: suite aggregate is not independently robust")
        report_count += verify_aggregate(aggregate_path, context)
        aggregate_paths.append(aggregate_path)
        aggregates[suite_id] = aggregate
        expected_files.add(Path(aggregate_relative))

    nominal_rate = f32(aggregates["nominal"]["pooledSuccessRate"])
    validation_rate = f32(
        aggregates["validation-collision"]["pooledSuccessRate"])
    require(f32(nominal_rate - validation_rate) <= f32(0.05),
            f"{manifest_path}: validation pooled success degraded by over 5pp")

    actual_files: set[Path] = set()
    actual_directories: set[Path] = {Path("qualification")}
    for path in qualification_directory.rglob("*"):
        require(not path.is_symlink(),
                f"{manifest_path}: qualification evidence contains a symlink")
        relative = path.relative_to(context.directory)
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise VerificationError(
                f"{manifest_path}: qualification contains a special file")
    require(actual_files == expected_files
            and actual_directories == {
                Path("qualification"), Path("qualification/nominal"),
                Path("qualification/validation-collision"),
            }, f"{manifest_path}: qualification inventory is not exact")
    expected_top_level = {
        "metadata.json", "policy.safetensors", "training-state.json",
        "deployment-manifest.json", "requalification-manifest.json",
        "qualification",
    }
    require({path.name for path in context.directory.iterdir()}
            == expected_top_level,
            f"{manifest_path}: bundle has missing or extra top-level entries")
    for name in expected_top_level:
        path = context.directory / name
        require(not path.is_symlink(),
                f"{manifest_path}: top-level entry is a symlink: {name}")
        require(path.is_dir() if name == "qualification" else path.is_file(),
                f"{manifest_path}: top-level entry has wrong type: {name}")
    verify_deployment_manifest(context, required=True, exact=True)
    return aggregate_paths, report_count


def verify(root: Path) -> tuple[int, int, int, int]:
    root = root.resolve()
    accepted, external_parity = catalog_entries(root)
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
        if evidence.get("schemaVersion") == 2:
            suite_aggregates, suite_reports = verify_schema2_requalification(
                root, context, evidence_path)
            aggregate_paths.update(path.resolve() for path in suite_aggregates)
            report_count += suite_reports
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
    for entry in external_parity:
        verify_external_parity(root, entry)
    return (len(accepted), len(external_parity), len(aggregate_paths),
            report_count)


def verify_app_checkpoint_package(root: Path, checkpoint_root: Path) -> None:
    """Require the distributable app to contain only selectable evidence.

    Historical and development checkpoints remain useful in a repository
    checkout, but shipping them would make the runtime package broader than
    the fail-closed replay catalog.  This exact inventory also catches a
    missing evidence file or an accidental recursive copy of ignored runs.
    """
    require(not checkpoint_root.is_symlink(),
            f"app checkpoint root must not be a symlink: {checkpoint_root}")
    checkpoint_root = checkpoint_root.resolve()
    require(checkpoint_root.is_dir(),
            f"app checkpoint root does not exist: {checkpoint_root}")
    h1_files = {
        "deployment-manifest.json",
        "metadata.json",
        "policy.safetensors",
        "requalification-manifest.json",
        "training-state.json",
        "qualification/aggregate.json",
        "qualification/eval-seed-51001.json",
        "qualification/eval-seed-51002.json",
        "qualification/eval-seed-51003.json",
        "qualification/eval-seed-51004.json",
    }
    unitree_files = {"LICENSE", "manifest.json", "policy.safetensors"}
    expected_files = {Path("README.md")}
    expected_files.update(
        Path(H1_REQUALIFIED_SELECTION) / relative for relative in h1_files)
    expected_files.update(
        Path("external/unitree-h1") / relative for relative in unitree_files)
    for profile in ARACHNE_QUALIFICATION_PROFILE.values():
        expected_files.update(
            Path(profile["checkpointRelativeDirectory"]) / relative
            for relative in arachne_candidate_files(profile)
        )

    actual_files: set[Path] = set()
    actual_directories: set[Path] = {Path(".")}
    for path in checkpoint_root.rglob("*"):
        require(not path.is_symlink(),
                f"app checkpoint package must not contain symlinks: {path}")
        relative = path.relative_to(checkpoint_root)
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise VerificationError(
                f"app checkpoint package has unsupported entry: {path}")

    expected_directories = {Path(".")}
    for relative in expected_files:
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent)
            parent = parent.parent
    require(actual_files == expected_files,
            "app checkpoint file inventory differs from the accepted/external "
            f"allowlist (missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)})")
    require(actual_directories == expected_directories,
            "app checkpoint directory inventory differs from the accepted/"
            f"external allowlist (missing="
            f"{sorted(expected_directories - actual_directories)}, extra="
            f"{sorted(actual_directories - expected_directories)})")

    source_root = root / "checkpoints"
    for relative in sorted(expected_files):
        packaged = checkpoint_root / relative
        source = source_root / relative
        require(source.is_file(),
                f"package allowlist source is missing: {source}")
        require(sha256_file(packaged) == sha256_file(source),
                f"packaged checkpoint bytes differ from tracked source: "
                f"{relative}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="verify accepted replay evidence and app packaging")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="repository root")
    parser.add_argument("--app-checkpoints", type=Path,
                        help="built app's checkpoints resource directory")
    arguments = parser.parse_args()
    root = arguments.root
    try:
        accepted, external_parity, aggregates, reports = verify(root)
        if arguments.app_checkpoints is not None:
            verify_app_checkpoint_package(root, arguments.app_checkpoints)
    except (VerificationError, KeyError, OverflowError, struct.error,
            TypeError, ValueError) as error:
        print(f"policy evidence verification failed: {error}", file=sys.stderr)
        return 1
    print(
        "verified policy evidence: "
        f"{accepted} accepted catalog entries, "
        f"{external_parity} external parity entries, "
        f"{aggregates} robustness aggregates, "
        f"{reports} evaluation reports"
        + (", exact app package" if arguments.app_checkpoints is not None
           else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
