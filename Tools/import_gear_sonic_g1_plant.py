#!/usr/bin/env python3
"""Import NVIDIA GEAR-SONIC's exact analytic G1 training plant as MJCF.

This tool intentionally supports one immutable upstream asset revision.  It
uses Apple's USD command-line tool only to turn the binary USD layers into a
small, auditable USDA subset; no USD Python package is required at runtime.
An optional, content-locked GEAR-SONIC deployment MJCF supplies render-only
STL meshes.  Those meshes are rigidly registered into the analytic plant's
zero-pose link frames and can never participate in contact or inertia.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORMAT = "avbd-gear-sonic-g1-analytic-plant-v1"
SOURCE_PROJECT = "NVLabs/GR00T-WholeBodyControl"
SOURCE_URL = "https://github.com/NVLabs/GR00T-WholeBodyControl"
SOURCE_REVISION = "60de0df7ffedeef415fe58d435e92cc5b01ba3d9"
SOURCE_LICENSE = "Apache-2.0"
USDCAT = Path("/usr/bin/usdcat")

EXPECTED_SOURCE_FILES = {
    "main_nodex_base.usd": (
        "09a030de1eb3ad527f310a00eb520ad99f8064dbf856797c8877be44e10df402"
    ),
    "main_nodex_physics.usd": (
        "26f15b6becc167e4f974b5ba5403118bb4c5b2d0c8857ab65fb253d4bb3c0d35"
    ),
}

EXPECTED_VISUAL_SOURCE_FILES = {
    "g1_29dof.xml": (
        "439c1ec0806583d73b492da9484b0cb9e9eae215e0d9506e3c2fa69016733532"
    ),
    "meshes/head_link.STL": (
        "d297328b3569fc85074355e76cba6c889d19de5361224bcd3c5b8ce82e13b9b6"
    ),
    "meshes/left_ankle_pitch_link.STL": (
        "d49e3abc6f5b12e532062cd575b87b5ef40cd2a3fc18f54a1ca5bba4f773d51d"
    ),
    "meshes/left_ankle_roll_link.STL": (
        "c4092af943141d4d9f74232f3cfa345afc6565f46a077793b8ae0e68b39dc33f"
    ),
    "meshes/left_elbow_link.STL": (
        "fa752198accd104d5c4c3a01382e45165b944fbbc5acce085059223324e5bed3"
    ),
    "meshes/left_hip_pitch_link.STL": (
        "4725168105ee768ee31638ef22b53f6be2d7641bfd7cfefe803488d884776fa4"
    ),
    "meshes/left_hip_roll_link.STL": (
        "91f25922ee8a7c3152790051bebad17b4d9cd243569c38fe340285ff93a97acf"
    ),
    "meshes/left_hip_yaw_link.STL": (
        "a16d88aa6ddac8083aa7ad55ed317bea44b1fa003d314fba88965b7ed0f3b55b"
    ),
    "meshes/left_knee_link.STL": (
        "8d92b9e3d3a636761150bb8025e32514c4602b91c7028d523ee42b3e632de477"
    ),
    "meshes/left_rubber_hand.STL": (
        "cff2221a690fa69303f61fce68f2d155c1517b52efb6ca9262dd56e0bc6e70fe"
    ),
    "meshes/left_shoulder_pitch_link.STL": (
        "f0d1cfd02fcf0d42f95e678eeca33da3afbcc366ffba5c052847773ec4f31d52"
    ),
    "meshes/left_shoulder_roll_link.STL": (
        "fb9df21687773522598dc384f1a2945c7519f11cbc8bd372a49170316d6eee88"
    ),
    "meshes/left_shoulder_yaw_link.STL": (
        "1aa97e9748e924336567992181f78c7cd0652fd52a4afcca3df6b2ef6f9e712e"
    ),
    "meshes/left_wrist_pitch_link.STL": (
        "b251d8e05047f695d0f536cd78c11973cfa4e78d08cfe82759336cc3471de3a9"
    ),
    "meshes/left_wrist_roll_link.STL": (
        "edc387c9a0ba8c2237e9b296d32531426fabeb6f53e58df45c76106bca74148c"
    ),
    "meshes/left_wrist_yaw_link.STL": (
        "83f8fb3a726bf9613d65dd14f0f447cb918c3c95b3938042a0c9c09749267d3b"
    ),
    "meshes/logo_link.STL": (
        "8571a0a19bc4916fa55f91449f51d5fdefd751000054865a842449429d5f155b"
    ),
    "meshes/pelvis.STL": (
        "5ba6bbc888e630550140d3c26763f10206da8c8bd30ed886b8ede41c61f57a31"
    ),
    "meshes/pelvis_contour_link.STL": (
        "5cc5c2c7a312329e3feeb2b03d3fc09fc29705bd01864f6767e51be959662420"
    ),
    "meshes/right_ankle_pitch_link.STL": (
        "15be426539ec1be70246d4d82a168806db64a41301af8b35c197a33348c787a9"
    ),
    "meshes/right_ankle_roll_link.STL": (
        "4b66222ea56653e627711b56d0a8949b4920da5df091da0ceb343f54e884e3a5"
    ),
    "meshes/right_elbow_link.STL": (
        "1be925d7aa268bb8fddf5362b9173066890c7d32092c05638608126e59d1e2ab"
    ),
    "meshes/right_hip_pitch_link.STL": (
        "e4f3c99d7f4a7d34eadbef9461fc66e3486cb5442db1ec50c86317d459f1a9c6"
    ),
    "meshes/right_hip_roll_link.STL": (
        "4c254ef66a356f492947f360dd931965477b631e3fcc841f91ccc46d945d54f6"
    ),
    "meshes/right_hip_yaw_link.STL": (
        "e479c2936ca2057e9eb2f7dff6c189b7419d7b8484dea0b298cbb36a2a6aa668"
    ),
    "meshes/right_knee_link.STL": (
        "63c4008449c9bbe701a6e2b557b7a252e90cf3a5abcf54cee46862b9a69f8ec8"
    ),
    "meshes/right_rubber_hand.STL": (
        "99533b778bca6246144fa511bb9d4e555e075c641f2a0251e04372869cd99d67"
    ),
    "meshes/right_shoulder_pitch_link.STL": (
        "24cdb387e0128dfe602770a81c56cdce3a0181d34d039a11d1aaf8819b7b8c02"
    ),
    "meshes/right_shoulder_roll_link.STL": (
        "962b97c48f9ce9e8399f45dd9522e0865d19aa9fd299406b2d475a8fc4a53e81"
    ),
    "meshes/right_shoulder_yaw_link.STL": (
        "a0b76489271da0c72461a344c9ffb0f0c6e64f019ea5014c1624886c442a2fe5"
    ),
    "meshes/right_wrist_pitch_link.STL": (
        "d22f8f3b3127f15a63e5be1ee273cd5075786c3142f1c3d9f76cbf43d2a26477"
    ),
    "meshes/right_wrist_roll_link.STL": (
        "a7ee9212ff5b94d6cb7f52bb1bbf3f352194d5b598acff74f4c77d340c5b344f"
    ),
    "meshes/right_wrist_yaw_link.STL": (
        "bc9dece2d12509707e0057ba2e48df8f3d56db0c79410212963a25e8a50f61a6"
    ),
    "meshes/torso_link.STL": (
        "e96d023f0368a4e3450b86ca5d4f10227d8141156a373e7da8cb3c93266523e0"
    ),
    "meshes/waist_roll_link.STL": (
        "34f0aa73f41131230be4d25876c944fdf6c24d62553f199ff8b980c15e8913df"
    ),
    "meshes/waist_support_link.STL": (
        "1fae9e1bb609848a1667d32eed8d6083ae443538a306843056a2a660f1b2926a"
    ),
    "meshes/waist_yaw_link.STL": (
        "2883f20e03f09b669b5b4ce10677ee6b5191c0934b584d7cbaef2d0662856ffb"
    ),
}

# The joint/drive declaration order in the USD is Unitree's hardware and the
# official GEAR-SONIC MuJoCo deployment order.  Policy tensor order is a
# separate contract handled by the policy importer.
JOINT_TREE = [
    ("left_hip_pitch_joint", "pelvis", "left_hip_pitch_link"),
    ("left_hip_roll_joint", "left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_yaw_joint", "left_hip_roll_link", "left_hip_yaw_link"),
    ("left_knee_joint", "left_hip_yaw_link", "left_knee_link"),
    ("left_ankle_pitch_joint", "left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_roll_joint", "left_ankle_pitch_link", "left_ankle_roll_link"),
    ("right_hip_pitch_joint", "pelvis", "right_hip_pitch_link"),
    ("right_hip_roll_joint", "right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_yaw_joint", "right_hip_roll_link", "right_hip_yaw_link"),
    ("right_knee_joint", "right_hip_yaw_link", "right_knee_link"),
    ("right_ankle_pitch_joint", "right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_roll_joint", "right_ankle_pitch_link", "right_ankle_roll_link"),
    ("waist_yaw_joint", "pelvis", "waist_yaw_link"),
    ("waist_roll_joint", "waist_yaw_link", "waist_roll_link"),
    ("waist_pitch_joint", "waist_roll_link", "torso_link"),
    ("left_shoulder_pitch_joint", "torso_link", "left_shoulder_pitch_link"),
    (
        "left_shoulder_roll_joint",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
    ),
    (
        "left_shoulder_yaw_joint",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
    ),
    ("left_elbow_joint", "left_shoulder_yaw_link", "left_elbow_link"),
    ("left_wrist_roll_joint", "left_elbow_link", "left_wrist_roll_link"),
    (
        "left_wrist_pitch_joint",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
    ),
    (
        "left_wrist_yaw_joint",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ),
    ("right_shoulder_pitch_joint", "torso_link", "right_shoulder_pitch_link"),
    (
        "right_shoulder_roll_joint",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
    ),
    (
        "right_shoulder_yaw_joint",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ),
    ("right_elbow_joint", "right_shoulder_yaw_link", "right_elbow_link"),
    ("right_wrist_roll_joint", "right_elbow_link", "right_wrist_roll_link"),
    (
        "right_wrist_pitch_joint",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
    ),
    (
        "right_wrist_yaw_joint",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ),
]

EXPECTED_MAX_FORCE = [
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
]

BODY_NAMES = ["pelvis", *[child for _, _, child in JOINT_TREE]]
JOINT_NAMES = [name for name, _, _ in JOINT_TREE]
PARENT_BY_CHILD = {child: parent for _, parent, child in JOINT_TREE}
JOINT_BY_CHILD = {child: name for name, _, child in JOINT_TREE}

EXPECTED_COLLIDER_COUNT = dict.fromkeys(BODY_NAMES, 0)
EXPECTED_COLLIDER_COUNT.update({
    "pelvis": 1,
    "left_hip_roll_link": 1,
    "left_knee_link": 1,
    "left_ankle_roll_link": 7,
    "right_hip_roll_link": 1,
    "right_knee_link": 1,
    "right_ankle_roll_link": 7,
    "torso_link": 4,
    "left_shoulder_yaw_link": 1,
    "left_elbow_link": 1,
    "left_wrist_yaw_link": 1,
    "right_shoulder_yaw_link": 1,
    "right_elbow_link": 1,
    "right_wrist_yaw_link": 1,
})

EXPECTED_VISUAL_COUNT = dict.fromkeys(BODY_NAMES, 1)
EXPECTED_VISUAL_COUNT.update({
    "pelvis": 2,
    "torso_link": 4,
    "left_wrist_yaw_link": 2,
    "right_wrist_yaw_link": 2,
})

AXES = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class Body:
    name: str
    center_of_mass: tuple[float, float, float]
    principal_axes: tuple[float, float, float, float]
    mass: float
    diagonal_inertia: tuple[float, float, float]


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    local_pos0: tuple[float, float, float]
    local_rot0: tuple[float, float, float, float]
    axis: tuple[float, float, float]
    lower_degrees: float
    upper_degrees: float
    max_force: float

    @property
    def lower_radians(self) -> float:
        return math.radians(self.lower_degrees)

    @property
    def upper_radians(self) -> float:
        return math.radians(self.upper_degrees)


@dataclass(frozen=True)
class Collider:
    body: str
    name: str
    kind: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    radius: float
    height: float | None


@dataclass(frozen=True)
class VisualMesh:
    name: str
    file: str
    source_path: Path


@dataclass(frozen=True)
class VisualGeometry:
    body: str
    mesh: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    rgba: tuple[float, float, float, float]


Transform = tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_revision(path: Path) -> str | None:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def require_close(
    actual: Iterable[float],
    expected: Iterable[float],
    context: str,
    tolerance: float = 2e-6,
) -> None:
    left = tuple(actual)
    right = tuple(expected)
    if len(left) != len(right) or any(
        not math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
        for a, b in zip(left, right)
    ):
        raise ValueError(f"{context}: got {left}, expected {right}")


def require_unit_quaternion(value: Iterable[float], context: str) -> None:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=2e-5):
        raise ValueError(f"{context}: non-unit quaternion (norm {norm})")


def _matching_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated USDA prim block")


def _blocks(text: str, pattern: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        opening = text.find("{", match.end())
        if opening < 0:
            raise ValueError(f"USDA declaration has no body: {match.group(0)}")
        closing = _matching_brace(text, opening)
        result.append((match.group(1), text[match.start():closing + 1]))
    return result


def _one_block(text: str, pattern: str, context: str) -> str:
    matches = _blocks(text, pattern)
    if len(matches) != 1:
        raise ValueError(f"{context}: expected one USDA block, found {len(matches)}")
    return matches[0][1]


def _single_match(block: str, pattern: str, context: str) -> str:
    matches = re.findall(pattern, block, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{context}: expected one value, found {len(matches)}")
    return matches[0]


def _numbers(value: str, count: int, context: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != count:
        raise ValueError(f"{context}: expected {count} components, found {parts}")
    result = tuple(float(part) for part in parts)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{context}: values must be finite")
    return result


def _tuple_attr(block: str, name: str, count: int, context: str) -> tuple[float, ...]:
    value = _single_match(
        block,
        rf"^[ \t]*(?:point3f|float3|double3|quatf|quatd)\s+"
        rf"{re.escape(name)}\s*=\s*\(([^)]*)\)",
        context,
    )
    return _numbers(value, count, context)


def _scalar_attr(block: str, name: str, context: str) -> float:
    value = _single_match(
        block,
        rf"^[ \t]*(?:float|double|int)\s+{re.escape(name)}\s*=\s*([^\s]+)",
        context,
    )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: value must be finite")
    return result


def _token_attr(block: str, name: str, context: str) -> str:
    return _single_match(
        block,
        rf'^[ \t]*(?:uniform\s+)?token\s+{re.escape(name)}\s*=\s*"([^"]+)"',
        context,
    )


def _relation_body(block: str, name: str, context: str) -> str:
    return _single_match(
        block,
        rf"^[ \t]*rel\s+{re.escape(name)}\s*=\s*</g1/([^>]+)>",
        context,
    )


def _validate_layer_metadata(text: str, context: str) -> None:
    if not re.search(r"\bmetersPerUnit\s*=\s*1(?:\.0+)?\b", text):
        raise ValueError(f"{context}: metersPerUnit must be 1")
    if not re.search(r'\bupAxis\s*=\s*"Z"', text):
        raise ValueError(f"{context}: upAxis must be Z")


def parse_physics(text: str) -> tuple[dict[str, Body], dict[str, Joint]]:
    _validate_layer_metadata(text, "physics layer")
    if "@main_nodex_base.usd@" not in text:
        raise ValueError("physics layer must sublayer main_nodex_base.usd")

    top_level_names = re.findall(r'^    over "([^"]+)"', text, re.MULTILINE)
    expected_top_level = {*BODY_NAMES, "joints"}
    if len(top_level_names) != 31 or set(top_level_names) != expected_top_level:
        raise ValueError(
            "physics layer must contain exactly 30 G1 body overlays and joints; "
            f"found {top_level_names}"
        )

    bodies: dict[str, Body] = {}
    for name in BODY_NAMES:
        block = _one_block(
            text,
            rf'^    over "({re.escape(name)})"(?:\s|\()',
            f"body {name}",
        )
        center = _tuple_attr(block, "physics:centerOfMass", 3, f"{name} COM")
        inertia = _tuple_attr(
            block, "physics:diagonalInertia", 3, f"{name} diagonal inertia"
        )
        mass = _scalar_attr(block, "physics:mass", f"{name} mass")
        axes = _tuple_attr(block, "physics:principalAxes", 4, f"{name} axes")
        if mass <= 0 or any(component <= 0 for component in inertia):
            raise ValueError(f"{name}: mass and principal inertia must be positive")
        require_unit_quaternion(axes, f"{name} principal axes")
        bodies[name] = Body(name, center, axes, mass, inertia)

    declared_joints = re.findall(
        r'^        def PhysicsRevoluteJoint "([^"]+)"', text, re.MULTILINE
    )
    if declared_joints != JOINT_NAMES:
        raise ValueError(
            "revolute joints are not in the exact G1 hardware order: "
            f"{declared_joints}"
        )

    joints: dict[str, Joint] = {}
    for index, (name, expected_parent, expected_child) in enumerate(JOINT_TREE):
        block = _one_block(
            text,
            rf'^        def PhysicsRevoluteJoint "({re.escape(name)})"(?:\s|\()',
            f"joint {name}",
        )
        if "PhysicsDriveAPI:angular" not in block:
            raise ValueError(f"{name}: missing angular drive actuator")
        parent = _relation_body(block, "physics:body0", f"{name} body0")
        child = _relation_body(block, "physics:body1", f"{name} body1")
        if (parent, child) != (expected_parent, expected_child):
            raise ValueError(
                f"{name}: parent tree is {(parent, child)}, expected "
                f"{(expected_parent, expected_child)}"
            )
        local_pos0 = _tuple_attr(block, "physics:localPos0", 3, f"{name} pos0")
        local_pos1 = _tuple_attr(block, "physics:localPos1", 3, f"{name} pos1")
        local_rot0 = _tuple_attr(block, "physics:localRot0", 4, f"{name} rot0")
        local_rot1 = _tuple_attr(block, "physics:localRot1", 4, f"{name} rot1")
        # This invariant is what permits the exact nested-MJCF simplification:
        # every child body frame is already the joint frame at q=0.
        require_close(local_pos1, (0, 0, 0), f"{name} localPos1")
        require_close(local_rot1, (1, 0, 0, 0), f"{name} localRot1")
        require_unit_quaternion(local_rot0, f"{name} localRot0")

        axis_name = _token_attr(block, "physics:axis", f"{name} axis")
        if axis_name not in AXES:
            raise ValueError(f"{name}: unsupported axis {axis_name}")
        lower = _scalar_attr(block, "physics:lowerLimit", f"{name} lower limit")
        upper = _scalar_attr(block, "physics:upperLimit", f"{name} upper limit")
        max_force = _scalar_attr(
            block, "drive:angular:physics:maxForce", f"{name} max force"
        )
        if lower >= upper or max_force <= 0:
            raise ValueError(f"{name}: invalid limits or actuator maxForce")
        if not math.isclose(max_force, EXPECTED_MAX_FORCE[index], abs_tol=1e-6):
            raise ValueError(
                f"{name}: maxForce {max_force} does not match exact asset "
                f"value {EXPECTED_MAX_FORCE[index]}"
            )
        for drive_name in (
            "drive:angular:physics:damping",
            "drive:angular:physics:stiffness",
            "drive:angular:physics:targetPosition",
        ):
            value = _scalar_attr(block, drive_name, f"{name} {drive_name}")
            if value != 0:
                raise ValueError(f"{name}: expected zero source drive {drive_name}")
        if _token_attr(
            block, "drive:angular:physics:type", f"{name} drive type"
        ) != "force":
            raise ValueError(f"{name}: angular drive must use force mode")
        joints[name] = Joint(
            name=name,
            parent=parent,
            child=child,
            local_pos0=local_pos0,
            local_rot0=local_rot0,
            axis=AXES[axis_name],
            lower_degrees=lower,
            upper_degrees=upper,
            max_force=max_force,
        )

    return bodies, joints


def _parse_extent(block: str, context: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = _single_match(
        block,
        r"^[ \t]*float3\[\]\s+extent\s*=\s*"
        r"\[\s*\(([^)]*)\)\s*,\s*\(([^)]*)\)\s*\]",
        context,
    )
    # re.findall returns a tuple when the expression has two capture groups.
    if not isinstance(values, tuple):
        raise ValueError(f"{context}: malformed extent")
    return _numbers(values[0], 3, context), _numbers(values[1], 3, context)


def parse_colliders(text: str) -> dict[str, list[Collider]]:
    _validate_layer_metadata(text, "collider layer")
    direct_bodies = re.findall(r'^    def Xform "([^"]+)"', text, re.MULTILINE)
    if len(direct_bodies) != 30 or set(direct_bodies) != set(BODY_NAMES):
        raise ValueError(
            "collider layer must contain exactly the 30 G1 bodies; "
            f"found {direct_bodies}"
        )

    result: dict[str, list[Collider]] = {}
    for body_name in BODY_NAMES:
        body_block = _one_block(
            text,
            rf'^    def Xform "({re.escape(body_name)})"\s*$',
            f"collider body {body_name}",
        )
        meshes = _blocks(body_block, r'^        def Xform "(mesh_\d+)"\s*$')
        expected_names = [f"mesh_{index}" for index in range(len(meshes))]
        if [name for name, _ in meshes] != expected_names:
            raise ValueError(f"{body_name}: collider proxy names are not sequential")
        if len(meshes) != EXPECTED_COLLIDER_COUNT[body_name]:
            raise ValueError(
                f"{body_name}: found {len(meshes)} colliders, expected "
                f"{EXPECTED_COLLIDER_COUNT[body_name]}"
            )

        colliders: list[Collider] = []
        for mesh_name, mesh_block in meshes:
            position = _tuple_attr(
                mesh_block, "xformOp:translate", 3, f"{body_name}/{mesh_name} pos"
            )
            rotation = _tuple_attr(
                mesh_block, "xformOp:orient", 4, f"{body_name}/{mesh_name} quat"
            )
            scale = _tuple_attr(
                mesh_block, "xformOp:scale", 3, f"{body_name}/{mesh_name} scale"
            )
            require_close(scale, (1, 1, 1), f"{body_name}/{mesh_name} scale")
            require_unit_quaternion(rotation, f"{body_name}/{mesh_name} rotation")
            xform_order = re.findall(
                r'"(xformOp:(?:translate|orient|scale))"',
                _single_match(
                    mesh_block,
                    r"^[ \t]*uniform token\[\]\s+xformOpOrder\s*=\s*\[([^]]*)\]",
                    f"{body_name}/{mesh_name} transform order",
                ),
            )
            if xform_order != [
                "xformOp:translate", "xformOp:orient", "xformOp:scale"
            ]:
                raise ValueError(f"{body_name}/{mesh_name}: unsupported transform order")

            primitives = _blocks(
                mesh_block, r'^            def (Sphere|Capsule) "[^"]+"\s*$'
            )
            if len(primitives) != 1:
                raise ValueError(
                    f"{body_name}/{mesh_name}: expected exactly one analytic primitive"
                )
            kind, primitive_block = primitives[0]
            radius = _scalar_attr(
                primitive_block, "radius", f"{body_name}/{mesh_name} radius"
            )
            if radius <= 0:
                raise ValueError(f"{body_name}/{mesh_name}: radius must be positive")
            lower_extent, upper_extent = _parse_extent(
                primitive_block, f"{body_name}/{mesh_name} extent"
            )
            if kind == "Sphere":
                height = None
                require_close(lower_extent, (-radius, -radius, -radius), "sphere extent")
                require_close(upper_extent, (radius, radius, radius), "sphere extent")
            else:
                if _token_attr(
                    primitive_block, "axis", f"{body_name}/{mesh_name} axis"
                ) != "Z":
                    raise ValueError(
                        f"{body_name}/{mesh_name}: exact asset capsule axis must be Z"
                    )
                height = _scalar_attr(
                    primitive_block, "height", f"{body_name}/{mesh_name} height"
                )
                if height <= 0:
                    raise ValueError(f"{body_name}/{mesh_name}: height must be positive")
                half_extent = height / 2 + radius
                require_close(
                    lower_extent, (-radius, -radius, -half_extent), "capsule extent"
                )
                require_close(
                    upper_extent, (radius, radius, half_extent), "capsule extent"
                )
            colliders.append(Collider(
                body=body_name,
                name=mesh_name,
                kind=kind.lower(),
                position=position,
                rotation=rotation,
                radius=radius,
                height=height,
            ))
        result[body_name] = colliders

    all_colliders = [collider for values in result.values() for collider in values]
    if len(all_colliders) != 29:
        raise ValueError(f"expected exactly 29 analytic colliders, found {len(all_colliders)}")
    if sum(collider.kind == "sphere" for collider in all_colliders) != 1:
        raise ValueError("exact collider asset must contain one sphere")
    if sum(collider.kind == "capsule" for collider in all_colliders) != 28:
        raise ValueError("exact collider asset must contain 28 capsules")
    return result


def _space_numbers(value: str, count: int, context: str) -> tuple[float, ...]:
    parts = value.split()
    if len(parts) != count:
        raise ValueError(f"{context}: expected {count} components, found {parts}")
    result = tuple(float(part) for part in parts)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{context}: values must be finite")
    return result


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quaternion_conjugate(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return value[0], -value[1], -value[2], -value[3]


def _normalized_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in value))
    if norm <= 1e-15:
        raise ValueError("zero quaternion")
    return tuple(component / norm for component in value)


def _rotate(
    rotation: tuple[float, float, float, float],
    value: tuple[float, float, float],
) -> tuple[float, float, float]:
    vector_quaternion = (0.0, *value)
    rotated = _quaternion_multiply(
        _quaternion_multiply(rotation, vector_quaternion),
        _quaternion_conjugate(rotation),
    )
    return rotated[1], rotated[2], rotated[3]


def _compose(left: Transform, right: Transform) -> Transform:
    rotated = _rotate(left[1], right[0])
    return (
        tuple(left[0][index] + rotated[index] for index in range(3)),
        _normalized_quaternion(_quaternion_multiply(left[1], right[1])),
    )


def _inverse(value: Transform) -> Transform:
    rotation = _quaternion_conjugate(_normalized_quaternion(value[1]))
    position = _rotate(rotation, tuple(-component for component in value[0]))
    return position, rotation


def _require_rotation_close(
    actual: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
    context: str,
    tolerance: float = 2e-9,
) -> None:
    direct = max(abs(a - b) for a, b in zip(actual, expected))
    negated = max(abs(a + b) for a, b in zip(actual, expected))
    if min(direct, negated) > tolerance:
        raise ValueError(f"{context}: got {actual}, expected {expected}")


def parse_visuals(
    model_path: Path,
    joints: dict[str, Joint],
) -> tuple[
    list[VisualMesh],
    dict[str, list[VisualGeometry]],
    dict[str, Transform],
]:
    try:
        root = ET.parse(model_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"visual source MJCF is malformed: {error}") from error
    if root.tag != "mujoco" or root.get("model") != "g1_29dof":
        raise ValueError("visual source must be NVIDIA's g1_29dof MJCF")
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian" \
            or compiler.get("meshdir") != "meshes":
        raise ValueError("visual source compiler must use radian and meshes/")

    asset = root.find("asset")
    if asset is None:
        raise ValueError("visual source is missing <asset>")
    mesh_nodes = asset.findall("mesh")
    if len(mesh_nodes) != 36 or len(asset) != 36:
        raise ValueError("visual source must declare exactly 36 mesh assets")
    meshes: list[VisualMesh] = []
    mesh_names: set[str] = set()
    mesh_files: set[str] = set()
    for node in mesh_nodes:
        name = node.get("name")
        file = node.get("file")
        if not name or not file or name in mesh_names or file in mesh_files:
            raise ValueError("visual mesh assets need unique names and files")
        if Path(file).name != file or Path(file).suffix != ".STL":
            raise ValueError(f"visual mesh {name} has unsafe or non-STL file {file}")
        if node.get("scale") not in (None, "1 1 1"):
            raise ValueError(f"visual mesh {name} uses unsupported scale")
        source_path = model_path.parent / "meshes" / file
        meshes.append(VisualMesh(name=name, file=file, source_path=source_path))
        mesh_names.add(name)
        mesh_files.add(file)

    worldbody = root.find("worldbody")
    if worldbody is None or len(worldbody.findall("body")) != 1:
        raise ValueError("visual source must have exactly one root body")
    root_body = worldbody.find("body")
    assert root_body is not None

    source_rest: dict[str, Transform] = {}
    raw_visuals: dict[str, list[VisualGeometry]] = {}
    body_order: list[str] = []

    def visit(node: ET.Element, parent: str | None) -> None:
        name = node.get("name")
        if not name or name in source_rest:
            raise ValueError("visual source body names must be present and unique")
        expected_parent = None if name == "pelvis" else PARENT_BY_CHILD.get(name)
        if parent != expected_parent:
            raise ValueError(
                f"visual source parent of {name} is {parent}, expected "
                f"{expected_parent}"
            )
        local_position = _space_numbers(
            node.get("pos", "0 0 0"), 3, f"visual body {name} position"
        )
        local_rotation = _space_numbers(
            node.get("quat", "1 0 0 0"), 4, f"visual body {name} rotation"
        )
        require_unit_quaternion(local_rotation, f"visual body {name} rotation")
        local_rotation = _normalized_quaternion(local_rotation)
        if name == "pelvis":
            require_close(
                local_position, (0, 0, 0.793), "visual source root position"
            )
            require_close(
                local_rotation, (1, 0, 0, 0), "visual source root rotation"
            )
            source_rest[name] = ((0, 0, 0), (1, 0, 0, 0))
            root_joints = node.findall("joint")
            if len(root_joints) != 1 \
                    or root_joints[0].get("name") != "floating_base_joint" \
                    or root_joints[0].get("type") != "free":
                raise ValueError("visual source root must have its exact free joint")
        else:
            assert parent is not None
            source_rest[name] = _compose(
                source_rest[parent], (local_position, local_rotation)
            )
            body_joints = node.findall("joint")
            if len(body_joints) != 1 \
                    or body_joints[0].get("name") != JOINT_BY_CHILD[name]:
                raise ValueError(f"visual source joint for {name} changed")
            require_close(
                _space_numbers(
                    body_joints[0].get("pos", "0 0 0"), 3,
                    f"visual source {name} joint position",
                ),
                (0, 0, 0),
                f"visual source {name} joint position",
            )

        body_visuals: list[VisualGeometry] = []
        for geom in node.findall("geom"):
            if geom.get("type") != "mesh":
                continue
            render_only = geom.get("contype") == "0" \
                and geom.get("conaffinity") == "0"
            if not render_only:
                # NVIDIA's deployment model also carries mesh/primitive
                # collision geometry. It is deliberately not imported: the
                # validated training USD remains the only physics authority.
                continue
            mesh = geom.get("mesh")
            if mesh not in mesh_names:
                raise ValueError(f"visual geom on {name} references {mesh}")
            if geom.get("density") != "0" or geom.get("group") != "1":
                raise ValueError(f"visual geom on {name} lost render-only metadata")
            position = _space_numbers(
                geom.get("pos", "0 0 0"), 3, f"visual geom {mesh} position"
            )
            rotation = _space_numbers(
                geom.get("quat", "1 0 0 0"), 4, f"visual geom {mesh} rotation"
            )
            rgba = _space_numbers(
                geom.get("rgba", "0.24 0.28 0.34 1"),
                4, f"visual geom {mesh} color",
            )
            require_unit_quaternion(rotation, f"visual geom {mesh} rotation")
            rotation = _normalized_quaternion(rotation)
            body_visuals.append(VisualGeometry(
                body=name, mesh=mesh, position=position,
                rotation=rotation, rgba=rgba,
            ))
        raw_visuals[name] = body_visuals
        body_order.append(name)
        for child in node.findall("body"):
            visit(child, name)

    visit(root_body, None)
    if body_order != BODY_NAMES:
        raise ValueError(
            f"visual source body order/tree changed: found {body_order}"
        )
    for name in BODY_NAMES:
        if len(raw_visuals[name]) != EXPECTED_VISUAL_COUNT[name]:
            raise ValueError(
                f"visual source {name} has {len(raw_visuals[name])} render "
                f"geoms, expected {EXPECTED_VISUAL_COUNT[name]}"
            )
    referenced_meshes = {
        visual.mesh for values in raw_visuals.values() for visual in values
    }
    if referenced_meshes != mesh_names:
        raise ValueError("visual source assets and render geoms are not one-to-one")

    training_rest: dict[str, Transform] = {
        "pelvis": ((0, 0, 0), (1, 0, 0, 0))
    }
    for joint_name, parent, child in JOINT_TREE:
        joint = joints[joint_name]
        training_rest[child] = _compose(
            training_rest[parent], (
                joint.local_pos0,
                _normalized_quaternion(joint.local_rot0),
            )
        )

    # The deployment MJCF and analytic training USD use the same semantic
    # links but differ in a few zero-pose frame translations (notably the
    # waist chain). Register each source link into the training link at the
    # exact authored zero pose; never silently assume coincident frames.
    alignments = {
        name: _compose(_inverse(training_rest[name]), source_rest[name])
        for name in BODY_NAMES
    }
    visuals: dict[str, list[VisualGeometry]] = {}
    for name in BODY_NAMES:
        mapped: list[VisualGeometry] = []
        for visual in raw_visuals[name]:
            position, rotation = _compose(
                alignments[name], (visual.position, visual.rotation)
            )
            expected_world = _compose(
                source_rest[name], (visual.position, visual.rotation)
            )
            actual_world = _compose(
                training_rest[name], (position, rotation)
            )
            require_close(
                actual_world[0], expected_world[0],
                f"visual {visual.mesh} zero-pose registration", tolerance=2e-9,
            )
            _require_rotation_close(
                actual_world[1], expected_world[1],
                f"visual {visual.mesh} zero-pose registration",
            )
            mapped.append(VisualGeometry(
                body=name, mesh=visual.mesh, position=position,
                rotation=rotation, rgba=visual.rgba,
            ))
        visuals[name] = mapped
    return meshes, visuals, alignments


def fmt(value: float) -> str:
    if value == 0 or abs(value) < 1e-18:
        return "0"
    return format(value, ".17g")


def fmt_values(values: Iterable[float]) -> str:
    return " ".join(fmt(value) for value in values)


def build_mjcf(
    bodies: dict[str, Body],
    joints: dict[str, Joint],
    colliders: dict[str, list[Collider]],
    visual_meshes: list[VisualMesh] | None = None,
    visuals: dict[str, list[VisualGeometry]] | None = None,
) -> str:
    root = ET.Element("mujoco", {"model": "gear_sonic_g1_29dof_cylinder"})
    if visual_meshes is None:
        root.append(ET.Comment(
            "Exact analytic GEAR-SONIC training plant; no visual or convex mesh geometry"
        ))
        ET.SubElement(root, "compiler", {"angle": "radian"})
    else:
        if visuals is None:
            raise ValueError("visual meshes require mapped visual geometry")
        root.append(ET.Comment(
            "Exact analytic GEAR-SONIC training plant with render-only source meshes"
        ))
        ET.SubElement(root, "compiler", {
            "angle": "radian", "meshdir": "meshes",
        })
        asset = ET.SubElement(root, "asset")
        for mesh in visual_meshes:
            ET.SubElement(asset, "mesh", {
                "name": mesh.name, "file": mesh.file,
            })
    world = ET.SubElement(root, "worldbody")

    children: dict[str, list[str]] = {name: [] for name in BODY_NAMES}
    for _, parent, child in JOINT_TREE:
        children[parent].append(child)

    def append_body(parent_xml: ET.Element, name: str) -> None:
        attributes = {"name": name}
        if name == "pelvis":
            # Root translation is deliberately an environment concern.  The
            # official training reset places this authored-zero root at z=.76.
            attributes["pos"] = "0 0 0"
            attributes["quat"] = "1 0 0 0"
        else:
            joint = joints[JOINT_BY_CHILD[name]]
            attributes["pos"] = fmt_values(joint.local_pos0)
            attributes["quat"] = fmt_values(joint.local_rot0)
        body_xml = ET.SubElement(parent_xml, "body", attributes)

        body = bodies[name]
        ET.SubElement(body_xml, "inertial", {
            "pos": fmt_values(body.center_of_mass),
            "quat": fmt_values(body.principal_axes),
            "mass": fmt(body.mass),
            "diaginertia": fmt_values(body.diagonal_inertia),
        })

        if name == "pelvis":
            ET.SubElement(body_xml, "freejoint", {"name": "floating_base_joint"})
        else:
            joint = joints[JOINT_BY_CHILD[name]]
            ET.SubElement(body_xml, "joint", {
                "name": joint.name,
                "type": "hinge",
                "pos": "0 0 0",
                "axis": fmt_values(joint.axis),
                "limited": "true",
                "range": fmt_values((joint.lower_radians, joint.upper_radians)),
                "actuatorfrclimited": "true",
                "actuatorfrcrange": fmt_values((-joint.max_force, joint.max_force)),
                "damping": "0",
            })

        for collider in colliders[name]:
            geom_attributes = {
                "name": f"{name}_{collider.name}",
                "type": collider.kind,
                "pos": fmt_values(collider.position),
                "quat": fmt_values(collider.rotation),
            }
            if collider.kind == "sphere":
                geom_attributes["size"] = fmt(collider.radius)
            else:
                assert collider.height is not None
                # MuJoCo capsule size is radius and cylinder half-height;
                # USD's height is the full cylinder length excluding caps.
                geom_attributes["size"] = fmt_values(
                    (collider.radius, collider.height / 2)
                )
            ET.SubElement(body_xml, "geom", geom_attributes)

        if visuals is not None:
            for visual in visuals[name]:
                ET.SubElement(body_xml, "geom", {
                    "name": f"visual_{visual.mesh}",
                    "type": "mesh",
                    "mesh": visual.mesh,
                    "pos": fmt_values(visual.position),
                    "quat": fmt_values(visual.rotation),
                    "rgba": fmt_values(visual.rgba),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                    "density": "0",
                })

        for child in children[name]:
            append_body(body_xml, child)

    append_body(world, "pelvis")

    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        ET.SubElement(actuator, "motor", {"name": name, "joint": name})

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"


def _xml_floats(element: ET.Element, attribute: str, count: int) -> tuple[float, ...]:
    value = element.get(attribute)
    if value is None:
        raise ValueError(f"generated <{element.tag}> lacks {attribute}")
    parts = value.split()
    if len(parts) != count:
        raise ValueError(f"generated <{element.tag}> {attribute} has wrong arity")
    return tuple(float(part) for part in parts)


def validate_generated_mjcf(
    xml_text: str,
    bodies: dict[str, Body],
    joints: dict[str, Joint],
    colliders: dict[str, list[Collider]],
    visual_meshes: list[VisualMesh] | None = None,
    visuals: dict[str, list[VisualGeometry]] | None = None,
) -> None:
    root = ET.fromstring(xml_text)
    if root.tag != "mujoco" or root.find("compiler").get("angle") != "radian":
        raise ValueError("generated plant is not radian MJCF")
    if visual_meshes is None:
        if root.find("asset") is not None or root.findall(".//mesh"):
            raise ValueError("generated analytic-only plant contains mesh assets")
        if root.find("compiler").get("meshdir") is not None:
            raise ValueError("generated analytic-only plant has a mesh directory")
    else:
        if visuals is None or root.find("compiler").get("meshdir") != "meshes":
            raise ValueError("generated visual plant lacks its mesh contract")
        generated_meshes = root.findall("./asset/mesh")
        if [(mesh.get("name"), mesh.get("file")) for mesh in generated_meshes] \
                != [(mesh.name, mesh.file) for mesh in visual_meshes]:
            raise ValueError("generated visual mesh assets changed")

    body_elements = root.findall(".//body")
    joint_elements = root.findall(".//joint")
    motor_elements = root.findall("./actuator/motor")
    geom_elements = root.findall(".//geom")
    visual_elements = [
        element for element in geom_elements
        if element.get("type") == "mesh"
        and element.get("contype") == "0"
        and element.get("conaffinity") == "0"
    ]
    collider_elements = [
        element for element in geom_elements if element not in visual_elements
    ]
    if [element.get("name") for element in body_elements] != BODY_NAMES:
        raise ValueError("generated MJCF body order changed")
    if [element.get("name") for element in joint_elements] != JOINT_NAMES:
        raise ValueError("generated MJCF joint order changed")
    if [element.get("name") for element in motor_elements] != JOINT_NAMES:
        raise ValueError("generated MJCF actuator order changed")
    if len(collider_elements) != 29:
        raise ValueError("generated MJCF must contain exactly 29 collision geoms")
    if any(
        element.get("type") not in {"sphere", "capsule"}
        for element in collider_elements
    ):
        raise ValueError("generated MJCF contains non-analytic collision geometry")
    expected_visual_count = 0 if visuals is None else sum(
        len(values) for values in visuals.values()
    )
    if len(visual_elements) != expected_visual_count:
        raise ValueError("generated MJCF visual geometry count changed")
    if any(
        element.get("density") != "0" or element.get("group") != "1"
        for element in visual_elements
    ):
        raise ValueError("generated visual mesh lost render-only metadata")

    element_by_body = {element.get("name"): element for element in body_elements}
    root_body = element_by_body["pelvis"]
    require_close(_xml_floats(root_body, "pos", 3), (0, 0, 0), "root position")
    if len(root_body.findall("freejoint")) != 1:
        raise ValueError("generated root must contain exactly one free joint")

    for body_name in BODY_NAMES:
        element = element_by_body[body_name]
        body = bodies[body_name]
        inertial = element.find("inertial")
        if inertial is None:
            raise ValueError(f"generated {body_name} lacks inertial")
        require_close(_xml_floats(inertial, "pos", 3), body.center_of_mass, "COM")
        require_close(_xml_floats(inertial, "quat", 4), body.principal_axes, "axes")
        require_close((float(inertial.get("mass")),), (body.mass,), "mass")
        require_close(
            _xml_floats(inertial, "diaginertia", 3),
            body.diagonal_inertia,
            "diagonal inertia",
        )
        if body_name != "pelvis":
            joint = joints[JOINT_BY_CHILD[body_name]]
            require_close(_xml_floats(element, "pos", 3), joint.local_pos0, "body pos")
            require_close(_xml_floats(element, "quat", 4), joint.local_rot0, "body quat")
            parent = next(
                ancestor
                for ancestor in body_elements
                if element in list(ancestor.findall("body"))
            )
            if parent.get("name") != joint.parent:
                raise ValueError(f"generated parent of {body_name} is wrong")
            joint_xml = element.find("joint")
            if joint_xml is None:
                raise ValueError(f"generated {body_name} lacks joint")
            require_close(_xml_floats(joint_xml, "axis", 3), joint.axis, "joint axis")
            require_close(
                _xml_floats(joint_xml, "range", 2),
                (joint.lower_radians, joint.upper_radians),
                "joint range",
            )
            require_close(
                _xml_floats(joint_xml, "actuatorfrcrange", 2),
                (-joint.max_force, joint.max_force),
                "actuator force range",
            )

        geom_by_name = {
            geom.get("name"): geom for geom in element.findall("geom")
            if geom.get("type") != "mesh"
        }
        expected_geoms = colliders[body_name]
        if len(geom_by_name) != len(expected_geoms):
            raise ValueError(f"generated collider count for {body_name} changed")
        for collider in expected_geoms:
            geom = geom_by_name[f"{body_name}_{collider.name}"]
            require_close(_xml_floats(geom, "pos", 3), collider.position, "geom pos")
            require_close(_xml_floats(geom, "quat", 4), collider.rotation, "geom quat")
            if collider.kind == "sphere":
                require_close(
                    _xml_floats(geom, "size", 1), (collider.radius,), "sphere size"
                )
            else:
                assert collider.height is not None
                require_close(
                    _xml_floats(geom, "size", 2),
                    (collider.radius, collider.height / 2),
                    "capsule size",
                )

        visual_by_name = {
            geom.get("name"): geom for geom in element.findall("geom")
            if geom.get("type") == "mesh"
        }
        expected_visuals = [] if visuals is None else visuals[body_name]
        if len(visual_by_name) != len(expected_visuals):
            raise ValueError(f"generated visual count for {body_name} changed")
        for visual in expected_visuals:
            geom = visual_by_name[f"visual_{visual.mesh}"]
            if geom.get("mesh") != visual.mesh \
                    or geom.get("contype") != "0" \
                    or geom.get("conaffinity") != "0":
                raise ValueError(f"generated visual {visual.mesh} is not render-only")
            require_close(
                _xml_floats(geom, "pos", 3), visual.position,
                f"visual {visual.mesh} position", tolerance=2e-9,
            )
            _require_rotation_close(
                _xml_floats(geom, "quat", 4), visual.rotation,
                f"visual {visual.mesh} rotation",
            )
            require_close(
                _xml_floats(geom, "rgba", 4), visual.rgba,
                f"visual {visual.mesh} color",
            )

    for motor, name in zip(motor_elements, JOINT_NAMES):
        if motor.get("joint") != name:
            raise ValueError(f"generated actuator {name} targets wrong joint")


def run_usdcat(base: Path, physics: Path) -> tuple[str, str, str]:
    if not USDCAT.is_file():
        raise SystemExit(f"Apple USD tool is unavailable: {USDCAT}")
    try:
        version = subprocess.check_output(
            [str(USDCAT), "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"failed to query {USDCAT}: {error}") from error

    with tempfile.TemporaryDirectory(prefix="avbd-gear-sonic-g1-") as temporary:
        directory = Path(temporary)
        physics_usda = directory / "physics.usda"
        colliders_usda = directory / "colliders.usda"
        commands = [
            [str(USDCAT), str(physics), "-o", str(physics_usda)],
            [
                str(USDCAT), "--flatten", "--mask", "/colliders",
                str(base), "-o", str(colliders_usda),
            ],
        ]
        for command in commands:
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise SystemExit(f"usdcat failed: {error}") from error
        return (
            physics_usda.read_text(encoding="utf-8"),
            colliders_usda.read_text(encoding="utf-8"),
            version,
        )


def validate_sources(base: Path, physics: Path) -> dict[str, str]:
    paths = {base.name: base, physics.name: physics}
    if set(paths) != set(EXPECTED_SOURCE_FILES):
        raise SystemExit(
            "expected main_nodex_base.usd and main_nodex_physics.usd, got "
            f"{sorted(paths)}"
        )
    hashes: dict[str, str] = {}
    for name, expected_hash in EXPECTED_SOURCE_FILES.items():
        path = paths[name]
        if not path.is_file():
            raise SystemExit(f"source does not exist: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"unsupported {name} SHA256 {actual_hash}; expected {expected_hash} "
                f"from {SOURCE_REVISION}"
            )
        hashes[name] = actual_hash
    detected_revision = git_revision(base)
    if detected_revision is not None and detected_revision != SOURCE_REVISION:
        raise SystemExit(
            f"source checkout is {detected_revision}; expected {SOURCE_REVISION}"
        )
    return hashes


def validate_visual_sources(model: Path) -> dict[str, str]:
    if model.name != "g1_29dof.xml" or not model.is_file():
        raise SystemExit(
            "visual source must be the pinned gear_sonic_deploy/g1/g1_29dof.xml"
        )
    hashes: dict[str, str] = {}
    for relative_path, expected_hash in EXPECTED_VISUAL_SOURCE_FILES.items():
        path = model if relative_path == "g1_29dof.xml" \
            else model.parent / relative_path
        if not path.is_file():
            raise SystemExit(f"visual source does not exist: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"unsupported visual {relative_path} SHA256 {actual_hash}; "
                f"expected {expected_hash} from {SOURCE_REVISION}"
            )
        hashes[relative_path] = actual_hash
    detected_revision = git_revision(model)
    if detected_revision is not None and detected_revision != SOURCE_REVISION:
        raise SystemExit(
            f"visual source checkout is {detected_revision}; "
            f"expected {SOURCE_REVISION}"
        )
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the exact NVIDIA GEAR-SONIC analytic G1 plant"
    )
    parser.add_argument("--base-usd", required=True, type=Path)
    parser.add_argument("--physics-usd", required=True, type=Path)
    parser.add_argument(
        "--visual-mjcf", type=Path,
        help=(
            "optional pinned gear_sonic_deploy/g1/g1_29dof.xml; its relative "
            "STLs are copied and emitted strictly as render-only geometry"
        ),
    )
    parser.add_argument("--output", "--output-dir", required=True, type=Path)
    args = parser.parse_args()

    base = args.base_usd.expanduser().resolve()
    physics = args.physics_usd.expanduser().resolve()
    visual_model = None if args.visual_mjcf is None \
        else args.visual_mjcf.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_hashes = validate_sources(base, physics)
    physics_text, colliders_text, usdcat_version = run_usdcat(base, physics)
    bodies, joints = parse_physics(physics_text)
    colliders = parse_colliders(colliders_text)
    visual_source_hashes: dict[str, str] | None = None
    visual_meshes: list[VisualMesh] | None = None
    visuals: dict[str, list[VisualGeometry]] | None = None
    visual_alignments: dict[str, Transform] | None = None
    if visual_model is not None:
        visual_source_hashes = validate_visual_sources(visual_model)
        visual_meshes, visuals, visual_alignments = parse_visuals(
            visual_model, joints
        )
        parsed_paths = {f"meshes/{mesh.file}" for mesh in visual_meshes}
        expected_paths = set(EXPECTED_VISUAL_SOURCE_FILES) - {"g1_29dof.xml"}
        if parsed_paths != expected_paths:
            raise SystemExit(
                "visual MJCF mesh declarations differ from the locked source files"
            )
    xml_text = build_mjcf(
        bodies, joints, colliders, visual_meshes, visuals
    )
    validate_generated_mjcf(
        xml_text, bodies, joints, colliders, visual_meshes, visuals
    )

    xml_bytes = xml_text.encode("utf-8")
    importer_path = Path(__file__).resolve()
    all_colliders = [value for values in colliders.values() for value in values]
    manifest = {
        "schemaVersion": 1,
        "format": FORMAT,
        "robot": "unitree-g1-29dof",
        "source": {
            "project": SOURCE_PROJECT,
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "files": {
                name: {"sha256": source_hashes[name]}
                for name in sorted(source_hashes)
            },
        },
        "importer": {
            "repositoryRevision": git_revision(importer_path),
            "scriptSHA256": sha256(importer_path),
            "usdcat": {
                "path": str(USDCAT),
                "version": usdcat_version,
                "commandContract": [
                    [str(USDCAT), "{physicsUsd}", "-o", "{temporaryPhysicsUsda}"],
                    [
                        str(USDCAT), "--flatten", "--mask", "/colliders",
                        "{baseUsd}", "-o", "{temporaryCollidersUsda}",
                    ],
                ],
            },
        },
        "plant": {
            "rootBody": "pelvis",
            "authoredRootPosition": [0, 0, 0],
            "environmentInitialRootPosition": [0, 0, 0.76],
            "bodyCount": len(bodies),
            "revoluteJointCount": len(joints),
            "actuatorCount": len(JOINT_NAMES),
            "collisionPrimitiveCount": len(all_colliders),
            "collisionPrimitiveKinds": {
                "sphere": sum(value.kind == "sphere" for value in all_colliders),
                "capsule": sum(value.kind == "capsule" for value in all_colliders),
            },
            "bodyNames": BODY_NAMES,
            "jointNamesInHardwareOrder": JOINT_NAMES,
            "parentByChild": PARENT_BY_CHILD,
            "actuatorMaxForceInHardwareOrder": [
                joints[name].max_force for name in JOINT_NAMES
            ],
            "collisionPrimitiveCountByBody": EXPECTED_COLLIDER_COUNT,
            "lengthUnit": "meter",
            "sourceJointLimitUnit": "degree",
            "mjcfJointLimitUnit": "radian",
            "collisionRepresentation": "source-authored analytic primitives",
            "containsVisualMeshes": visual_meshes is not None,
            "containsFabricatedConvexMeshes": False,
        },
        "output": {
            "plantXML": "plant.xml",
            "plantXMLSHA256": sha256_bytes(xml_bytes),
        },
    }

    if visual_meshes is not None:
        assert visual_source_hashes is not None
        assert visuals is not None
        assert visual_alignments is not None
        manifest["visualSource"] = {
            "project": SOURCE_PROJECT,
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "model": "gear_sonic_deploy/g1/g1_29dof.xml",
            "files": {
                name: {"sha256": visual_source_hashes[name]}
                for name in sorted(visual_source_hashes)
            },
        }
        manifest["plant"].update({
            "visualMeshAssetCount": len(visual_meshes),
            "visualGeometryCount": sum(
                len(values) for values in visuals.values()
            ),
            "visualGeometryCountByBody": EXPECTED_VISUAL_COUNT,
            "visualRepresentation": "source-authored render-only STL meshes",
            "visualCollisionSemantics": {
                "contype": 0,
                "conaffinity": 0,
                "density": 0,
            },
            "visualFrameMapping": (
                "exact source-to-training zero-pose rigid registration"
            ),
            "visualFrameAlignmentByBody": {
                name: {
                    "position": list(visual_alignments[name][0]),
                    "quaternionWXYZ": list(visual_alignments[name][1]),
                }
                for name in BODY_NAMES
            },
        })
        manifest["output"]["visualMeshes"] = {
            f"meshes/{mesh.file}": {
                "sha256": visual_source_hashes[f"meshes/{mesh.file}"]
            }
            for mesh in visual_meshes
        }

    output.mkdir(parents=True, exist_ok=True)
    if visual_meshes is not None:
        mesh_output = output / "meshes"
        mesh_output.mkdir(parents=True, exist_ok=True)
        for mesh in visual_meshes:
            destination = mesh_output / mesh.file
            shutil.copyfile(mesh.source_path, destination)
            expected_hash = visual_source_hashes[f"meshes/{mesh.file}"]
            if sha256(destination) != expected_hash:
                raise SystemExit(f"copied visual mesh {mesh.file} changed bytes")
    (output / "plant.xml").write_bytes(xml_bytes)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "plant_manifest.json").write_text(manifest_text, encoding="utf-8")

    # Validate persisted bytes as the final guard against accidental encoding
    # or path-handling changes.
    if sha256(output / "plant.xml") != manifest["output"]["plantXMLSHA256"]:
        raise SystemExit("persisted plant.xml hash does not match manifest")
    loaded = json.loads((output / "plant_manifest.json").read_text(encoding="utf-8"))
    if loaded != manifest:
        raise SystemExit("persisted plant manifest failed round-trip validation")

    print(
        f"imported GEAR-SONIC G1 plant: {len(bodies)} bodies, "
        f"{len(joints)} revolute joints, {len(JOINT_NAMES)} actuators, "
        f"{len(all_colliders)} analytic colliders, "
        f"{0 if visual_meshes is None else len(visual_meshes)} render meshes"
    )
    print(f"plant.xml SHA256 {manifest['output']['plantXMLSHA256']}")


if __name__ == "__main__":
    main()
