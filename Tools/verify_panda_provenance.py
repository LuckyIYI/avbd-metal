#!/usr/bin/env python3
"""Verify the Panda plant and ManiSkill PushT attribution boundaries.

The default check is hermetic and validates every redistributed payload plus
the independently authored pusher contract. Optional source checkouts prove
the hashes against the two immutable upstream revisions without networking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROOT = ROOT / "Development"
PANDA_MANIFEST = (
    DEVELOPMENT_ROOT / "Sources/Robotics/Assets/panda_pusher/PROVENANCE.json"
)
MANISKILL_MANIFEST = (
    DEVELOPMENT_ROOT / "Sources/Robotics/Assets/maniskill_pusht/PROVENANCE.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    if data.get("schemaVersion") != 1:
        raise VerificationError(f"unsupported schema in {path}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_file(path: Path, expected: str, label: str) -> None:
    require(SHA256_PATTERN.fullmatch(expected) is not None,
            f"{label} has an invalid SHA-256")
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256(path)
    require(actual == expected,
            f"{label} hash mismatch: expected {expected}, got {actual}")


def verify_upstream(manifest: dict[str, Any], checkout: Path | None,
                    expected_repository: str, label: str) -> None:
    upstream = manifest["upstream"]
    require(upstream["repository"] == expected_repository,
            f"{label} repository is not the approved primary source")
    require(REVISION_PATTERN.fullmatch(upstream["revision"]) is not None,
            f"{label} source revision is not an immutable commit")
    require(upstream["license"] == "Apache-2.0",
            f"{label} source license changed")
    require(upstream["noticeFile"] is None,
            f"{label} manifest must identify any upstream NOTICE")
    paths: set[str] = set()
    for source in upstream["files"]:
        path = source["path"]
        require(path not in paths, f"duplicate {label} source path: {path}")
        paths.add(path)
        require(SHA256_PATTERN.fullmatch(source["sha256"]) is not None,
                f"{label} source {path} has an invalid SHA-256")
        if checkout is not None:
            verify_file(checkout / path, source["sha256"],
                        f"{label} source {path}")


def floats(value: str | None, default: str) -> list[float]:
    return [float(item) for item in (value or default).split()]


def normalized(values: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in values))
    require(magnitude > 0, "zero quaternion in Panda source")
    return [value / magnitude for value in values]


def quaternion_matrix(values: list[float]) -> list[list[float]]:
    w, x, y, z = normalized(values)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ]


def close_vectors(actual: list[float], expected: list[float],
                  label: str, tolerance: float = 1e-9) -> None:
    require(len(actual) == len(expected), f"{label} length changed")
    error = max((abs(a - b) for a, b in zip(actual, expected)), default=0)
    require(error <= tolerance, f"{label} changed (maximum error {error})")


def body_map(root: ET.Element) -> tuple[dict[str, ET.Element], dict[str, str | None]]:
    bodies: dict[str, ET.Element] = {}
    parents: dict[str, str | None] = {}

    def visit(body: ET.Element, parent: str | None) -> None:
        name = body.attrib["name"]
        require(name not in bodies, f"duplicate Panda body {name}")
        bodies[name] = body
        parents[name] = parent
        for child in body.findall("./body"):
            visit(child, name)

    worldbody = root.find("./worldbody")
    require(worldbody is not None, "missing Panda worldbody")
    for body in worldbody.findall("./body"):
        visit(body, None)
    return bodies, parents


def verify_inertia(source: ET.Element, reduced: ET.Element,
                   body_name: str) -> None:
    source_inertial = source.find("./inertial")
    reduced_inertial = reduced.find("./inertial")
    require(source_inertial is not None and reduced_inertial is not None,
            f"missing {body_name} inertial")
    require(abs(float(source_inertial.attrib["mass"])
                - float(reduced_inertial.attrib["mass"])) <= 1e-9,
            f"{body_name} mass differs from Menagerie")
    close_vectors(floats(reduced_inertial.get("pos"), "0 0 0"),
                  floats(source_inertial.get("pos"), "0 0 0"),
                  f"{body_name} center of mass")

    full = floats(source_inertial.get("fullinertia"), "")
    require(len(full) == 6, f"{body_name} source full inertia is missing")
    source_tensor = [
        [full[0], full[3], full[4]],
        [full[3], full[1], full[5]],
        [full[4], full[5], full[2]],
    ]
    diagonal = floats(reduced_inertial.get("diaginertia"), "")
    rotation = quaternion_matrix(floats(
        reduced_inertial.get("quat"), "1 0 0 0"))
    reconstructed = [
        [sum(rotation[row][axis] * diagonal[axis]
             * rotation[column][axis] for axis in range(3))
         for column in range(3)]
        for row in range(3)
    ]
    error = max(abs(source_tensor[row][column]
                    - reconstructed[row][column])
                for row in range(3) for column in range(3))
    require(error <= 2e-9,
            f"{body_name} inertia diagonalization changed (error {error})")


def stl_bounds(path: Path) -> tuple[list[float], list[float]]:
    payload = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if payload[:5].lower() == b"solid" and b"facet" in payload[:100]:
        for line in payload.decode(errors="ignore").splitlines():
            values = line.split()
            if len(values) == 4 and values[0] == "vertex":
                vertices.append(tuple(float(value) for value in values[1:]))
    else:
        require(len(payload) >= 84, f"invalid STL: {path}")
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        require(len(payload) >= 84 + triangle_count * 50,
                f"truncated STL: {path}")
        for triangle in range(triangle_count):
            start = 84 + triangle * 50 + 12
            for vertex in range(3):
                vertices.append(struct.unpack_from(
                    "<fff", payload, start + vertex * 12))
    require(bool(vertices), f"empty STL: {path}")
    return ([min(vertex[axis] for vertex in vertices) for axis in range(3)],
            [max(vertex[axis] for vertex in vertices) for axis in range(3)])


def verify_panda_transform(source_path: Path, reduced_path: Path,
                           source_checkout: Path) -> None:
    source_root = ET.parse(source_path).getroot()
    reduced_root = ET.parse(reduced_path).getroot()
    source_bodies, source_parents = body_map(source_root)
    reduced_bodies, reduced_parents = body_map(reduced_root)
    expected_names = [f"link{index}" for index in range(8)]
    require(list(reduced_bodies) == expected_names,
            "reduced Panda body set/order differs from link0...link7")
    for name in expected_names:
        require(source_parents[name] == reduced_parents[name],
                f"{name} parent differs from Menagerie")
        close_vectors(floats(reduced_bodies[name].get("pos"), "0 0 0"),
                      floats(source_bodies[name].get("pos"), "0 0 0"),
                      f"{name} frame position")
        reduced_quaternion = normalized(floats(
            reduced_bodies[name].get("quat"), "1 0 0 0"))
        source_quaternion = normalized(floats(
            source_bodies[name].get("quat"), "1 0 0 0"))
        if sum(a * b for a, b in zip(reduced_quaternion,
                                     source_quaternion)) < 0:
            reduced_quaternion = [-value for value in reduced_quaternion]
        close_vectors(reduced_quaternion, source_quaternion,
                      f"{name} frame rotation")
        verify_inertia(source_bodies[name], reduced_bodies[name], name)

    source_joint_default = source_root.find(
        "./default/default[@class='panda']/joint")
    reduced_joint_default = reduced_root.find(
        "./default/default[@class='panda']/joint")
    require(source_joint_default is not None
            and reduced_joint_default is not None,
            "missing Panda joint defaults")
    for attribute, default in (("axis", "0 0 1"),
                               ("range", "-2.8973 2.8973")):
        close_vectors(floats(reduced_joint_default.get(attribute), default),
                      floats(source_joint_default.get(attribute), default),
                      f"Panda default joint {attribute}")
    for attribute, default in (("armature", "0"), ("damping", "0")):
        close_vectors(floats(reduced_joint_default.get(attribute), default),
                      floats(source_joint_default.get(attribute), default),
                      f"Panda default joint {attribute}")
    for index in range(1, 8):
        name = f"link{index}"
        source_joint = source_bodies[name].find("./joint")
        reduced_joint = reduced_bodies[name].find("./joint")
        require(source_joint is not None and reduced_joint is not None,
                f"missing joint{index}")
        require(source_joint.get("name") == reduced_joint.get("name")
                == f"joint{index}", f"joint{index} binding changed")
        close_vectors(
            floats(reduced_joint.get("range"),
                   reduced_joint_default.attrib["range"]),
            floats(source_joint.get("range"),
                   source_joint_default.attrib["range"]),
            f"joint{index} range")

    source_general_default = source_root.find(
        "./default/default[@class='panda']/general")
    require(source_general_default is not None,
            "missing Menagerie Panda actuator default")
    source_generals = source_root.findall("./actuator/general")[:7]
    reduced_motors = reduced_root.findall("./actuator/motor")
    require(len(source_generals) == len(reduced_motors) == 7,
            "Panda arm actuator count changed")
    for source, reduced in zip(source_generals, reduced_motors):
        require(source.get("name") == reduced.get("name"),
                "Panda actuator name changed")
        require(source.get("joint") == reduced.get("joint"),
                f"{source.get('name')} joint binding changed")
        expected_range = floats(
            source.get("forcerange"), source_general_default.attrib["forcerange"])
        close_vectors(floats(reduced.get("forcerange"), ""), expected_range,
                      f"{source.get('name')} effort limit")

    link0_geom = reduced_bodies["link0"].find("./geom")
    require(link0_geom is not None and link0_geom.get("type") == "box",
            "link0 proxy is not the audited AABB box")
    center = floats(link0_geom.get("pos"), "0 0 0")
    half_size = floats(link0_geom.get("size"), "")
    source_minimum, source_maximum = stl_bounds(
        source_checkout / "franka_emika_panda/assets/link0.stl")
    proxy_minimum = [center[axis] - half_size[axis] for axis in range(3)]
    proxy_maximum = [center[axis] + half_size[axis] for axis in range(3)]
    require(all(proxy_minimum[axis] <= source_minimum[axis]
                and proxy_maximum[axis] >= source_maximum[axis]
                for axis in range(2))
            and proxy_maximum[2] >= source_maximum[2]
            and 0 <= proxy_minimum[2] <= 1e-7
            and proxy_minimum[2] - source_minimum[2] < 5e-5,
            "link0 proxy no longer matches the audited, plane-clamped AABB")
    require(max(source_minimum[axis] - proxy_minimum[axis]
                for axis in range(3)) < 0.001
            and max(proxy_maximum[axis] - source_maximum[axis]
                    for axis in range(3)) < 0.001,
            "link0 proxy is no longer a tight outward-rounded AABB")


def verify_panda(checkout: Path | None) -> None:
    manifest = load_manifest(PANDA_MANIFEST)
    verify_upstream(
        manifest, checkout,
        "https://github.com/google-deepmind/mujoco_menagerie",
        "Menagerie Panda")
    asset = manifest["asset"]
    asset_path = DEVELOPMENT_ROOT / asset["path"]
    verify_file(asset_path, asset["sha256"], "Panda pusher MJCF")
    if checkout is not None:
        verify_panda_transform(
            checkout / "franka_emika_panda/panda.xml",
            asset_path, checkout)
    license_data = manifest["redistributedLicense"]
    verify_file(DEVELOPMENT_ROOT / license_data["path"], license_data["sha256"],
                "Menagerie Panda license")

    root = ET.parse(asset_path).getroot()
    require(root.tag == "mujoco" and root.get("model") == asset["modelName"],
            "Panda pusher MJCF model identity changed")
    require(root.get("model") == "avbd_panda_pusher",
            "Panda pusher MJCF must not retain the old asset identity")
    pusher = root.find(".//geom[@name='avbd_pusher']")
    require(pusher is not None, "missing first-party AVBD pusher geometry")
    require(pusher.get("type") == "capsule", "pusher must remain a capsule")
    size = [float(value) for value in pusher.attrib["size"].split()]
    position = [float(value) for value in pusher.attrib["pos"].split()]
    design = manifest["firstPartyPusher"]
    radius = size[0]
    half_segment = size[1]
    half_envelope = radius + half_segment
    require(abs(radius - design["radiusMeters"]) < 1e-12,
            "pusher radius differs from first-party design record")
    require(abs((2 * half_envelope) - design["totalLengthMeters"]) < 1e-12,
            "pusher length differs from first-party design record")
    require(abs((position[2] - half_envelope)
                - design["proximalEndpointMeters"]) < 1e-12,
            "pusher proximal endpoint differs from design record")
    require(abs((position[2] + half_envelope)
                - design["distalEndpointAndTCPMeters"]) < 1e-12,
            "pusher TCP differs from design record")

    panda_default = root.find("./default/default[@class='panda']/geom")
    require(panda_default is not None,
            "missing explicit Menagerie/MuJoCo geom defaults")
    require(panda_default.get("friction") == "1 0.005 0.0001",
            "Panda proxy friction no longer materializes MuJoCo defaults")
    motors = root.findall("./actuator/motor")
    require(len(motors) == 7, "Panda plant must expose seven arm motors")
    efforts = [abs(float(motor.attrib["forcerange"].split()[1]))
               for motor in motors]
    require(efforts == [87.0] * 4 + [12.0] * 3,
            "Panda effort limits differ from the pinned Menagerie plant")


def verify_maniskill(checkout: Path | None) -> None:
    manifest = load_manifest(MANISKILL_MANIFEST)
    verify_upstream(
        manifest, checkout, "https://github.com/haosulab/ManiSkill",
        "ManiSkill PushT")
    for redistributed in manifest["redistributedFiles"]:
        verify_file(DEVELOPMENT_ROOT / redistributed["path"], redistributed["sha256"],
                    f"ManiSkill attribution {Path(redistributed['path']).name}")
    adapted = manifest["adaptedFile"]
    verify_file(DEVELOPMENT_ROOT / adapted["path"], adapted["sha256"],
                "adapted ManiSkill PushT task")
    require(manifest["assetBoundary"]["redistributesManiSkillAssets"] is False,
            "ManiSkill asset boundary must remain explicit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--menagerie-checkout", type=Path,
        help="checkout rooted at the pinned MuJoCo Menagerie revision")
    parser.add_argument(
        "--maniskill-checkout", type=Path,
        help="checkout rooted at the pinned ManiSkill revision")
    args = parser.parse_args()
    try:
        verify_panda(args.menagerie_checkout)
        verify_maniskill(args.maniskill_checkout)
    except VerificationError as error:
        print(f"Panda provenance verification failed: {error}", file=sys.stderr)
        return 1
    print("Panda provenance verified: Menagerie plant, AVBD pusher, "
          "and ManiSkill task attribution are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
