#!/usr/bin/env python3
"""Verify AVBD's Isaac Lab H1 configuration attribution boundary.

The default check is hermetic: it validates the pinned manifest, bundled
BSD-3-Clause license, authored notice, local adaptation markers and absence of
unlisted payloads. ``--source-dir`` optionally proves the recorded SHA-256
values against an independently obtained checkout of the immutable revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIRECTORY = ROOT / "Sources/AVBDCore/Assets/isaac_lab_h1"
MANIFEST_PATH = ASSET_DIRECTORY / "PROVENANCE.json"

EXPECTED_REPOSITORY = "https://github.com/isaac-sim/IsaacLab"
EXPECTED_RELEASE = "v2.3.2"
EXPECTED_REVISION = "37ddf626871758333d6ed89cf64ad702aef127d0"
EXPECTED_LICENSE = "BSD-3-Clause"
EXPECTED_UPSTREAM_FILES = {
    "LICENSE",
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/flat_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/rsl_rl_ppo_cfg.py",
}
EXPECTED_IMPLEMENTATION_FILES = {
    "Sources/AVBDCore/Robotics/HumanoidWalkTask.swift",
    "Sources/AVBDCore/Robotics/HumanoidIsaacVelocityTask.swift",
}
EXPECTED_CONTRACTS = {
    "h1-control-profile",
    "h1-flat-velocity-task",
    "h1-flat-ppo-reference",
}
EXPECTED_ASSET_FILES = {"LICENSE", "NOTICE", "PROVENANCE.json"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> None:
    require(SHA256_PATTERN.fullmatch(expected) is not None,
            f"{label} has an invalid SHA-256")
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256(path)
    require(actual == expected,
            f"{label} hash mismatch: expected {expected}, got {actual}")


def load_manifest() -> dict[str, Any]:
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"cannot read {MANIFEST_PATH}: {error}") from error
    require(data.get("schemaVersion") == 1,
            "unsupported Isaac Lab H1 provenance schema")
    return data


def normalized(text: str) -> str:
    return " ".join(text.split())


def verify_upstream(manifest: dict[str, Any],
                    source_directory: Path | None) -> dict[str, str]:
    upstream = manifest.get("upstream")
    require(isinstance(upstream, dict), "missing upstream object")
    require(upstream.get("project") == "Isaac Lab",
            "upstream project changed")
    require(upstream.get("repository") == EXPECTED_REPOSITORY,
            "upstream repository is not the approved primary source")
    require(upstream.get("release") == EXPECTED_RELEASE,
            "Isaac Lab release changed")
    revision = upstream.get("revision")
    require(revision == EXPECTED_REVISION,
            "Isaac Lab revision changed")
    require(isinstance(revision, str)
            and REVISION_PATTERN.fullmatch(revision) is not None,
            "Isaac Lab revision is not an immutable commit")
    require(upstream.get("license") == EXPECTED_LICENSE,
            "Isaac Lab source license changed")
    require(upstream.get("noticeFile") is None,
            "manifest must identify any upstream NOTICE file")

    entries = upstream.get("files")
    require(isinstance(entries, list), "upstream files must be a list")
    hashes: dict[str, str] = {}
    for entry in entries:
        require(isinstance(entry, dict), "invalid upstream file entry")
        path = entry.get("path")
        expected_hash = entry.get("sha256")
        require(isinstance(path, str) and path not in hashes,
                f"invalid or duplicate upstream path: {path}")
        require(isinstance(expected_hash, str)
                and SHA256_PATTERN.fullmatch(expected_hash) is not None,
                f"invalid upstream hash for {path}")
        hashes[path] = expected_hash
        if source_directory is not None:
            verify_file(source_directory / path, expected_hash,
                        f"Isaac Lab source {path}")
    require(set(hashes) == EXPECTED_UPSTREAM_FILES,
            "consulted Isaac Lab source set changed without verifier review")
    return hashes


def verify_adaptation(manifest: dict[str, Any],
                      upstream_paths: set[str]) -> None:
    adaptation = manifest.get("adaptation")
    require(isinstance(adaptation, dict), "missing adaptation object")
    require(adaptation.get("exactSourceEquivalence") is False,
            "provenance must not claim exact Isaac Lab equivalence")
    statement = adaptation.get("statement")
    require(isinstance(statement, str)
            and "not an exact PhysX environment or policy port" in statement,
            "adaptation statement does not preserve the non-equivalence boundary")

    implementations = adaptation.get("implementationFiles")
    require(isinstance(implementations, list),
            "implementationFiles must be a list")
    implementation_paths: set[str] = set()
    for implementation in implementations:
        require(isinstance(implementation, dict),
                "invalid implementation file entry")
        path = implementation.get("path")
        role = implementation.get("role")
        require(isinstance(path, str) and path not in implementation_paths,
                f"invalid or duplicate implementation path: {path}")
        require(isinstance(role, str) and role.strip(),
                f"missing adaptation role for {path}")
        require((ROOT / path).is_file(), f"missing implementation file: {path}")
        implementation_paths.add(path)
    require(implementation_paths == EXPECTED_IMPLEMENTATION_FILES,
            "Isaac Lab adaptation file set changed without verifier review")

    contracts = manifest.get("contracts")
    require(isinstance(contracts, list), "contracts must be a list")
    contract_ids: set[str] = set()
    referenced_sources: set[str] = set()
    for contract in contracts:
        require(isinstance(contract, dict), "invalid contract entry")
        contract_id = contract.get("id")
        require(isinstance(contract_id, str) and contract_id not in contract_ids,
                f"invalid or duplicate contract id: {contract_id}")
        contract_ids.add(contract_id)
        source_files = contract.get("sourceFiles")
        require(isinstance(source_files, list) and source_files,
                f"{contract_id} has no source files")
        for path in source_files:
            require(path in upstream_paths and path != "LICENSE",
                    f"{contract_id} references unpinned source {path}")
            referenced_sources.add(path)
        for field in ("adaptedConcepts", "avbdDifferences"):
            values = contract.get(field)
            require(isinstance(values, list) and values
                    and all(isinstance(value, str) and value.strip()
                            for value in values),
                    f"{contract_id} has an invalid {field} list")
    require(contract_ids == EXPECTED_CONTRACTS,
            "adaptation contract set changed without verifier review")
    require(referenced_sources == upstream_paths - {"LICENSE"},
            "not every consulted Isaac Lab source is assigned to a contract")


def verify_local_markers(manifest: dict[str, Any]) -> None:
    groups = manifest.get("localContractMarkers")
    require(isinstance(groups, list) and groups,
            "localContractMarkers must be a non-empty list")
    covered_paths: set[str] = set()
    for group in groups:
        require(isinstance(group, dict), "invalid marker group")
        path = group.get("path")
        markers = group.get("markers")
        require(path in EXPECTED_IMPLEMENTATION_FILES
                and path not in covered_paths,
                f"invalid or duplicate marker path: {path}")
        require(isinstance(markers, list) and markers,
                f"{path} has no contract markers")
        source = normalized((ROOT / path).read_text())
        for marker in markers:
            require(isinstance(marker, str) and marker.strip(),
                    f"{path} contains an empty marker")
            canonical_marker = normalized(marker)
            require(canonical_marker in source,
                    f"Isaac Lab adaptation marker changed in {path}: {marker}")
        covered_paths.add(path)
    require(covered_paths == EXPECTED_IMPLEMENTATION_FILES,
            "not every adapted implementation has contract markers")


def verify_redistribution(manifest: dict[str, Any],
                          upstream_hashes: dict[str, str]) -> None:
    boundary = manifest.get("redistributionBoundary")
    require(isinstance(boundary, dict)
            and boundary.get("redistributesIsaacLabSourceOrAssets") is False,
            "redistribution boundary changed")
    statement = boundary.get("statement")
    require(isinstance(statement, str)
            and "no upstream source, USD, mesh, texture or checkpoint payload"
            in statement,
            "redistribution boundary statement is incomplete")

    entries = manifest.get("redistributedFiles")
    require(isinstance(entries, list), "redistributedFiles must be a list")
    redistributed: dict[str, str] = {}
    for entry in entries:
        require(isinstance(entry, dict), "invalid redistributed file entry")
        path = entry.get("path")
        expected_hash = entry.get("sha256")
        require(isinstance(path, str) and path not in redistributed,
                f"invalid or duplicate redistributed path: {path}")
        require(isinstance(expected_hash, str),
                f"missing redistributed hash for {path}")
        verify_file(ROOT / path, expected_hash, f"redistributed file {path}")
        redistributed[path] = expected_hash

    license_path = "Sources/AVBDCore/Assets/isaac_lab_h1/LICENSE"
    notice_path = "Sources/AVBDCore/Assets/isaac_lab_h1/NOTICE"
    require(set(redistributed) == {license_path, notice_path},
            "redistributed Isaac Lab attribution file set changed")
    require(redistributed[license_path] == upstream_hashes["LICENSE"],
            "bundled BSD license is not byte-identical to pinned upstream")

    actual_assets = {
        path.name for path in ASSET_DIRECTORY.iterdir() if path.is_file()
    }
    require(actual_assets == EXPECTED_ASSET_FILES,
            "unreviewed payload exists in Isaac Lab H1 attribution directory")

    notice = (ROOT / notice_path).read_text()
    for required in (
        EXPECTED_REPOSITORY,
        EXPECTED_RELEASE,
        EXPECTED_REVISION,
        "This is an adaptation, not an exact Isaac Lab environment or policy port.",
        "No Isaac Lab USD, mesh, texture, policy checkpoint, Python source file",
    ):
        require(required in notice,
                f"Isaac Lab H1 notice is missing: {required}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        help=("optional Isaac Lab checkout rooted at pinned revision; verifies "
              "every consulted upstream file hash without network access"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        manifest = load_manifest()
        upstream_hashes = verify_upstream(
            manifest, arguments.source_dir)
        verify_adaptation(manifest, set(upstream_hashes))
        verify_local_markers(manifest)
        verify_redistribution(manifest, upstream_hashes)
    except (OSError, VerificationError) as error:
        print(f"Isaac Lab H1 provenance verification failed: {error}",
              file=sys.stderr)
        return 1
    source_suffix = " and pinned source checkout" \
        if arguments.source_dir is not None else ""
    print(
        "verified Isaac Lab v2.3.2 H1 attribution, local adaptation "
        f"contracts, BSD license{source_suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
