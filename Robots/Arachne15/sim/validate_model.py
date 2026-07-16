#!/usr/bin/env python3
"""Dependency-free structural validation for generated Arachne MJCF assets."""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import generate_model as model


ROOT = Path(__file__).resolve().parent


def floats(value: str) -> list[float]:
    return [float(token) for token in value.split()]


def validate(path: Path, expected_profile: str) -> dict:
    root = ET.parse(path).getroot()
    assert root.tag == "mujoco"
    assert root.attrib["model"] == f"arachne15_{expected_profile}"

    bodies = root.findall("./worldbody//body")
    joints = root.findall("./worldbody//joint")
    motors = root.findall("./actuator/motor")
    inertials = root.findall(".//body/inertial")
    exclusions = root.findall("./contact/exclude")
    assert len(bodies) == 17, len(bodies)
    assert len(joints) == 16, len(joints)
    assert len(motors) == 16, len(motors)
    assert len(inertials) == 17, len(inertials)
    assert len(exclusions) == 16, len(exclusions)

    names = [body.attrib["name"] for body in bodies]
    assert len(names) == len(set(names))
    joint_names = [joint.attrib["name"] for joint in joints]
    assert len(joint_names) == len(set(joint_names))
    assert {motor.attrib["joint"] for motor in motors} == set(joint_names)

    total_mass = 0.0
    for inertial in inertials:
        mass = float(inertial.attrib["mass"])
        diagonal = floats(inertial.attrib["diaginertia"])
        assert mass > 0 and len(diagonal) == 3 and min(diagonal) > 0
        # Necessary physical-inertia triangle inequalities.
        assert diagonal[0] + diagonal[1] >= diagonal[2] - 1e-12
        assert diagonal[0] + diagonal[2] >= diagonal[1] - 1e-12
        assert diagonal[1] + diagonal[2] >= diagonal[0] - 1e-12
        total_mass += mass
    assert math.isclose(total_mass, 1.220, abs_tol=2e-6), total_mass

    visual_meshes = []
    collision_geoms = []
    for geom in root.findall("./worldbody//geom"):
        if geom.attrib.get("type") == "mesh":
            assert geom.attrib.get("class") == "visual"
            visual_meshes.append(geom)
        elif geom.attrib.get("name") != "floor" and geom.attrib.get("class") != "visual":
            assert geom.attrib.get("type") in {"box", "capsule", "sphere"}
            collision_geoms.append(geom)
    assert visual_meshes
    assert not any(geom.attrib.get("type") == "mesh" for geom in collision_geoms)

    compiler = root.find("compiler")
    assert compiler is not None
    mesh_dir = (path.parent / compiler.attrib["meshdir"]).resolve()
    for mesh in root.findall("./asset/mesh"):
        assert floats(mesh.attrib["scale"]) == [0.001, 0.001, 0.001]
        assert (mesh_dir / mesh.attrib["file"]).is_file(), mesh.attrib["file"]

    for motor in motors:
        low, high = floats(motor.attrib["ctrlrange"])
        assert math.isclose(low, -model.DESIGN_TORQUE, abs_tol=1e-9)
        assert math.isclose(high, model.DESIGN_TORQUE, abs_tol=1e-9)

    foot_colliders = [
        geom for geom in collision_geoms
        if geom.attrib.get("name", "").endswith("_foot_collision")
    ]
    assert len(foot_colliders) == 8
    for geom in foot_colliders:
        assert geom.attrib["type"] == "box"
        assert floats(geom.attrib["size"]) == list(model.FOOT_HALF_SIZE)

    foot_bottom = (
        model.ROOT_WORLD_Z + model.HIP_Z
        - model.TIBIA_LENGTH * math.sin(model.TIBIA_PITCH)
        - model.FOOT_VERTICAL_SUPPORT
    )
    assert abs(foot_bottom) < 1e-10, foot_bottom

    return {
        "profile": expected_profile,
        "bodies": len(bodies),
        "joints": len(joints),
        "actuators": len(motors),
        "collision_geoms": len(collision_geoms),
        "visual_mesh_instances": len(visual_meshes),
        "total_mass_kg": total_mass,
        "nominal_foot_bottom_m": foot_bottom,
    }


def main() -> None:
    reports = [
        validate(ROOT / "arachne15_training.xml", "training"),
        validate(ROOT / "arachne15_validation.xml", "validation"),
        validate(model.BUNDLED_ASSET_ROOT / "arachne15_training.xml", "training"),
        validate(model.BUNDLED_ASSET_ROOT / "arachne15_validation.xml", "validation"),
    ]
    assert reports[0]["collision_geoms"] < reports[1]["collision_geoms"]
    assert reports[0] == reports[2]
    assert reports[1] == reports[3]
    for report in reports:
        fields = " ".join(f"{key}={value}" for key, value in report.items())
        print("PASS", fields)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ET.ParseError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(2)
