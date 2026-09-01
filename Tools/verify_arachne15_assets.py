#!/usr/bin/env python3
"""Hermetically verify the packaged Arachne-15 runtime asset snapshot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
import sys
import xml.etree.ElementTree as ET
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_RELATIVE_DIRECTORY = Path(
    "Development/Sources/Robotics/Assets/arachne15"
)
EXPECTED_FILE_SHA256: dict[str, str] = {
    "README.md":
        "7bc9552ab39cfd29e5ce926f43ecf75016455667fd7cc6f35a5dc56f470cdf63",
    "arachne15_training.xml":
        "880e42dbb672bdcea798c3538d5d44eb7b73082e43c8a073ba572dc21bb7f38f",
    "arachne15_validation.xml":
        "8eddeb8250791cdb8f53bee7816ab06dde2dd8f80768e71bace45984c70fd959",
    "meshes/battery_cradle.stl":
        "4291cc1a590447d9e635713a77e7263396dcdcecdc40a8111d7361f4e0436f8a",
    "meshes/chassis.stl":
        "4d55a9531c4b562fdb2a4008d9fc96751f5267c374572356677f743c3251f702",
    "meshes/coxa_link.stl":
        "f71587d549f00e5fb2d7eaa2e24ce939c32df9642078c7d5c3d987f7f4c85319",
    "meshes/foot_pad.stl":
        "9333a9130848ed20caf50f1fd47397fc93d1bd15b9e4632b44be001df05f3d04",
    "meshes/phone_guide.stl":
        "9dd0fcd3abb6ebcc4e74217d8166be89b3310452dc83cd095bfc1bf79811bd70",
    "meshes/tibia_link.stl":
        "d96e667332570a93fe647857d487db5df777d01282c198c04348113e69a22f4f",
}
EXPECTED_MESHES = {
    "battery_cradle.stl", "chassis.stl", "coxa_link.stl", "foot_pad.stl",
    "phone_guide.stl", "tibia_link.stl",
}


class VerificationError(RuntimeError):
    """A packaged-asset invariant was violated."""


@dataclass(frozen=True)
class ProfileReport:
    profile: str
    bodies: int
    joints: int
    actuators: int
    collisions: int
    visual_meshes: int
    mass_kg: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[set[Path], set[Path]]:
    try:
        mode = root.lstat().st_mode
    except OSError as error:
        raise VerificationError(f"asset root is unavailable: {error}") from error
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
            f"asset root must be a real directory: {root}")
    files, directories = set(), {Path(".")}

    def visit(directory: Path, relative: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            child = relative / entry.name
            mode = entry.stat(follow_symlinks=False).st_mode
            require(not stat.S_ISLNK(mode),
                    f"asset tree contains a symlink: {child}")
            if stat.S_ISDIR(mode):
                directories.add(child)
                visit(Path(entry.path), child)
            elif stat.S_ISREG(mode):
                files.add(child)
            else:
                raise VerificationError(
                    f"asset tree contains a special file: {child}")

    visit(root, Path("."))
    return files, directories


def verify_tree(root: Path, expected_hashes: Mapping[str, str]) -> None:
    expected = {Path(path) for path in expected_hashes}
    files, directories = inventory(root)
    require(files == expected,
            "Arachne asset file inventory changed "
            f"(missing={sorted(expected - files)}, extra={sorted(files - expected)})")
    require(directories == {Path("."), Path("meshes")},
            "Arachne asset directory inventory changed")
    for relative, digest in expected_hashes.items():
        require(len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"malformed SHA-256 for {relative}")
        actual = sha256_file(root / relative)
        require(actual == digest,
                f"Arachne asset SHA-256 changed for {relative}: "
                f"expected {digest}, got {actual}")


def verify_binary_stl(path: Path) -> None:
    data = path.read_bytes()
    require(len(data) >= 84, f"binary STL is truncated: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    require(triangles > 0 and len(data) == 84 + 50 * triangles,
            f"binary STL length disagrees with its triangle count: {path}")


def floats(value: str | None, label: str) -> tuple[float, ...]:
    require(value is not None, f"missing {label}")
    try:
        result = tuple(float(token) for token in value.split())
    except ValueError as error:
        raise VerificationError(f"non-numeric {label}") from error
    require(all(math.isfinite(item) for item in result), f"non-finite {label}")
    return result


def verify_profile(path: Path, profile: str) -> ProfileReport:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise VerificationError(f"cannot parse {path}: {error}") from error
    require(root.tag == "mujoco"
            and root.get("model") == f"arachne15_{profile}",
            f"{path}: model identity changed")
    compiler = root.find("./compiler")
    require(compiler is not None and compiler.get("meshdir") == "meshes",
            f"{path}: packaged mesh directory changed")

    declared_meshes = root.findall("./asset/mesh")
    filenames = {mesh.get("file") for mesh in declared_meshes}
    for mesh in declared_meshes:
        filename = str(mesh.get("file"))
        require(Path(filename).name == filename,
                f"{path}: mesh file escapes the packaged mesh directory")
        require(floats(mesh.get("scale"), "mesh scale") == (0.001, 0.001, 0.001),
                f"{path}: mesh scale changed")
        require((path.parent / "meshes" / filename).is_file(),
                f"{path}: missing referenced mesh {filename}")
    require(len(declared_meshes) == 6 and filenames == EXPECTED_MESHES,
            f"{path}: mesh declarations changed")

    bodies = root.findall("./worldbody//body")
    joints = root.findall("./worldbody//joint")
    motors = root.findall("./actuator/motor")
    require(len(bodies) == 17
            and len({body.get("name") for body in bodies}) == 17,
            f"{path}: body inventory changed")
    joint_names = {joint.get("name") for joint in joints}
    require(len(joints) == len(joint_names) == 16,
            f"{path}: joint inventory changed")
    require(len(root.findall("./worldbody//freejoint")) == 1,
            f"{path}: floating-base articulation changed")
    require(len(motors) == 16
            and {motor.get("joint") for motor in motors} == joint_names,
            f"{path}: actuator-to-joint mapping changed")
    for motor in motors:
        require(floats(motor.get("ctrlrange"), "motor control range")
                == (-0.186, 0.186),
                f"{path}: motor torque contract changed")

    inertials = root.findall("./worldbody//body/inertial")
    require(len(inertials) == 17
            and all(len(body.findall("./inertial")) == 1 for body in bodies),
            f"{path}: each body must have one inertial")
    total_mass = 0.0
    for inertial in inertials:
        mass = floats(inertial.get("mass"), "body mass")
        diagonal = floats(inertial.get("diaginertia"), "diagonal inertia")
        require(len(mass) == 1 and mass[0] > 0
                and len(diagonal) == 3 and min(diagonal) > 0
                and diagonal[0] + diagonal[1] >= diagonal[2] - 1e-12
                and diagonal[0] + diagonal[2] >= diagonal[1] - 1e-12
                and diagonal[1] + diagonal[2] >= diagonal[0] - 1e-12,
                f"{path}: invalid mass/inertia")
        total_mass += mass[0]
    require(math.isclose(total_mass, 1.220, rel_tol=0, abs_tol=2e-6),
            f"{path}: total mass changed: {total_mass}")

    geoms = root.findall("./worldbody//geom")
    mesh_visuals = [geom for geom in geoms if geom.get("type") == "mesh"]
    require(len(mesh_visuals) == 28
            and all(geom.get("class") == "visual" for geom in mesh_visuals),
            f"{path}: mesh geometry must remain visual-only")
    collisions = [
        geom for geom in geoms
        if geom.get("name") != "floor" and geom.get("class") != "visual"
    ]
    require(len(collisions) == {"training": 39, "validation": 60}[profile]
            and all(geom.get("type") in {"box", "capsule", "sphere"}
                    for geom in collisions),
            f"{path}: primitive collision inventory changed")
    require(len(root.findall("./contact/exclude")) == 16,
            f"{path}: adjacent-body exclusions changed")
    return ProfileReport(
        profile, len(bodies), len(joints), len(motors), len(collisions),
        len(mesh_visuals), total_mass,
    )


def verify_asset_directory(
    directory: Path,
    *,
    expected_hashes: Mapping[str, str] = EXPECTED_FILE_SHA256,
) -> tuple[ProfileReport, ProfileReport]:
    verify_tree(directory, expected_hashes)
    for relative in expected_hashes:
        if relative.endswith(".stl"):
            verify_binary_stl(directory / relative)
    reports = tuple(
        verify_profile(directory / f"arachne15_{profile}.xml", profile)
        for profile in ("training", "validation")
    )
    require(reports[0].bodies == reports[1].bodies
            and reports[0].joints == reports[1].joints
            and reports[0].actuators == reports[1].actuators
            and math.isclose(reports[0].mass_kg, reports[1].mass_kg),
            "training and validation articulation contracts differ")
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    try:
        reports = verify_asset_directory(args.root / ASSET_RELATIVE_DIRECTORY)
    except (VerificationError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 2
    for report in reports:
        print(
            "PASS", f"profile={report.profile}", f"bodies={report.bodies}",
            f"joints={report.joints}", f"actuators={report.actuators}",
            f"collisions={report.collisions}",
            f"visual_meshes={report.visual_meshes}",
            f"mass_kg={report.mass_kg:.3f}",
        )
    print(f"PASS pinned_files={len(EXPECTED_FILE_SHA256)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
