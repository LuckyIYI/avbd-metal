#!/usr/bin/env python3
"""Generate deterministic MuJoCo/AVBD assets for the Arachne-15 robot.

The printable STL meshes are visual geometry only.  Contact uses explicit
primitive compounds so the training and validation models have deterministic,
GPU-friendly collision topology instead of renderer-dependent mesh cooking.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
BUNDLED_ASSET_ROOT = REPOSITORY_ROOT / "Sources/AVBDCore/Assets/arachne15"
VISUAL_MESH_FILES = (
    "chassis.stl", "phone_guide.stl", "battery_cradle.stl",
    "coxa_link.stl", "tibia_link.stl", "foot_pad.stl",
)

HIP_X = (-0.060, -0.024, 0.024, 0.060)
HIP_Y = 0.092
HIP_Z = 0.022
COXA_LENGTH = 0.050
TIBIA_LENGTH = 0.105
TIBIA_PITCH = math.radians(65.0)
# Conservative half-extents of the printable TPU pad. The previous spherical
# contact represented only its thickness; when the tibia pitched, the much
# longer pad visibly passed through the floor. An oriented box preserves the
# physical support envelope and remains deterministic and GPU-friendly.
FOOT_HALF_SIZE = (0.00875, 0.00700, 0.00400)
FOOT_VERTICAL_SUPPORT = (
    FOOT_HALF_SIZE[0] * math.sin(TIBIA_PITCH)
    + FOOT_HALF_SIZE[2] * math.cos(TIBIA_PITCH)
)
ROOT_WORLD_Z = (
    FOOT_VERTICAL_SUPPORT
    + TIBIA_LENGTH * math.sin(TIBIA_PITCH)
    - HIP_Z
)

PHONE_SIZE = (0.00825, 0.14660, 0.07060)
PHONE_CENTER = (0.0, 0.0, 0.002 + PHONE_SIZE[2] / 2)
GUIDE_Y = 0.14780 / 2 + 0.008 / 2

DESIGN_TORQUE = 0.186  # 20% of the XC330-M288-T 5 V stall torque.


@dataclass(frozen=True)
class BoxMass:
    mass: float
    size: tuple[float, float, float]
    center: tuple[float, float, float]


def mass_properties(parts: list[BoxMass]) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    mass = sum(p.mass for p in parts)
    center = tuple(sum(p.mass * p.center[i] for p in parts) / mass for i in range(3))
    inertia = [0.0, 0.0, 0.0]
    for part in parts:
        sx, sy, sz = part.size
        own = (
            part.mass * (sy * sy + sz * sz) / 12,
            part.mass * (sx * sx + sz * sz) / 12,
            part.mass * (sx * sx + sy * sy) / 12,
        )
        dx = part.center[0] - center[0]
        dy = part.center[1] - center[1]
        dz = part.center[2] - center[2]
        inertia[0] += own[0] + part.mass * (dy * dy + dz * dz)
        inertia[1] += own[1] + part.mass * (dx * dx + dz * dz)
        inertia[2] += own[2] + part.mass * (dx * dx + dy * dy)
    return mass, center, tuple(inertia)


def root_mass_parts() -> list[BoxMass]:
    parts = [
        BoxMass(0.187000, PHONE_SIZE, PHONE_CENTER),
        # Printed root pieces: chassis includes the conservative rounding
        # margin that closes the 280 g whole-print load budget.
        BoxMass(0.123229, (0.154, 0.212, 0.007), (0, 0, 0.0035)),
        BoxMass(0.009847, (0.022, 0.008, 0.052), (0, GUIDE_Y, 0.026)),
        BoxMass(0.009847, (0.022, 0.008, 0.052), (0, -GUIDE_Y, 0.026)),
        BoxMass(0.000938, (0.014, 0.009, 0.012), (-0.008, GUIDE_Y, 0.048)),
        BoxMass(0.000938, (0.014, 0.009, 0.012), (-0.008, -GUIDE_Y, 0.048)),
        BoxMass(0.014681, (0.040, 0.131, 0.004), (0, 0, -0.026)),
        BoxMass(0.135000, (0.035, 0.105, 0.018), (0, 0, -0.020)),
        BoxMass(0.045000, (0.055, 0.030, 0.014), (-0.034, 0, -0.015)),
        BoxMass(0.025000, (0.055, 0.025, 0.012), (0.024, 0, -0.014)),
        BoxMass(0.013000, (0.130, 0.160, 0.006), (0, 0, 0.000)),
        BoxMass(0.055000, (0.130, 0.160, 0.006), (0, 0, -0.005)),
    ]
    # Hip motor stators and their root-side frames belong to the base link.
    for x in HIP_X:
        for y in (-HIP_Y, HIP_Y):
            parts.append(BoxMass(0.023, (0.020, 0.029, 0.034), (x, y, 0.005)))
            parts.append(BoxMass(0.003, (0.028, 0.028, 0.007), (x, y, 0.0035)))
    return parts


COXA_PARTS = [
    # The knee motor stator moves with the coxa; the hip stator does not.
    BoxMass(0.023000, (0.020, 0.029, 0.034), (0.050, 0, 0)),
    BoxMass(0.004548, (0.068, 0.018, 0.007), (0.025, 0, 0)),
    BoxMass(0.005000, (0.025, 0.025, 0.015), (0.050, 0, 0)),
    BoxMass(0.003000, (0.050, 0.005, 0.005), (0.025, 0, -0.005)),
]

TIBIA_PARTS = [
    BoxMass(0.009343, (0.120, 0.016, 0.009), (0.0525, 0, 0)),
    BoxMass(0.001174, (0.01746, 0.01397, 0.00762), (0.105, 0, 0)),
    BoxMass(0.001000, (0.014, 0.014, 0.010), (0, 0, 0)),
    BoxMass(0.002000, (0.095, 0.004, 0.004), (0.0525, 0, 0.005)),
]


def fmt(values) -> str:
    if isinstance(values, (int, float)):
        values = (values,)
    return " ".join(f"{value:.9g}" for value in values)


def indent(lines: list[str], levels: int = 1) -> list[str]:
    prefix = "  " * levels
    return [prefix + line for line in lines]


def box_geom(name: str, size, pos, *, friction: float = 0.55) -> str:
    half = tuple(value / 2 for value in size)
    return (
        f'<geom name="{name}" class="collision" type="box" '
        f'size="{fmt(half)}" pos="{fmt(pos)}" friction="{friction:.3g} 0.005 0.0001"/>'
    )


def visual_box(name: str, size, pos, *, rgba: str = "0.22 0.25 0.30 1",
               quat: str | None = None) -> str:
    half = tuple(value / 2 for value in size)
    rotation = f' quat="{quat}"' if quat else ""
    return (
        f'<geom name="{name}" class="visual" type="box" '
        f'size="{fmt(half)}" pos="{fmt(pos)}"{rotation} rgba="{rgba}"/>'
    )


def root_visual_components() -> list[str]:
    geoms = []
    for i, x in enumerate(HIP_X):
        for side, y in (("right", -HIP_Y), ("left", HIP_Y)):
            geoms.append(visual_box(
                f"hip_servo_{side}_{i}_visual", (0.020, 0.029, 0.034),
                (x, y, 0.005), rgba="0.16 0.18 0.22 1"))
    geoms += [
        visual_box("battery_visual", (0.035, 0.105, 0.018),
                   (0, 0, -0.020), rgba="0.10 0.12 0.16 1"),
        visual_box("bec_visual", (0.055, 0.030, 0.014),
                   (-0.034, 0, -0.015), rgba="0.10 0.26 0.42 1"),
        visual_box("control_bridge_visual", (0.055, 0.025, 0.012),
                   (0.024, 0, -0.014), rgba="0.10 0.38 0.24 1"),
    ]
    return geoms


def root_collision_geoms(profile: str) -> list[str]:
    geoms = [
        # Closed ring represented as four rails plus the structural phone spine.
        box_geom("base_x_rear", (0.014, 0.176, 0.005), (-0.070, 0, 0.0025)),
        box_geom("base_x_front", (0.014, 0.176, 0.005), (0.070, 0, 0.0025)),
        box_geom("base_y_right", (0.126, 0.014, 0.005), (0, -0.081, 0.0025)),
        box_geom("base_y_left", (0.126, 0.014, 0.005), (0, 0.081, 0.0025)),
        box_geom("phone_spine", (0.028, 0.160, 0.007), (0, 0, 0.0035)),
        box_geom("iphone_body", PHONE_SIZE, PHONE_CENTER, friction=0.35),
        box_geom("battery", (0.035, 0.105, 0.018), (0, 0, -0.020)),
    ]
    if profile == "validation":
        for i, x in enumerate(HIP_X):
            for side, y in (("right", -HIP_Y), ("left", HIP_Y)):
                geoms.append(box_geom(f"hip_pod_{side}_{i}", (0.028, 0.028, 0.007),
                                      (x, y, 0.0035)))
                geoms.append(box_geom(f"hip_servo_{side}_{i}", (0.020, 0.029, 0.034),
                                      (x, y, 0.005)))
        geoms += [
            box_geom("guide_left", (0.022, 0.008, 0.052), (0, GUIDE_Y, 0.026)),
            box_geom("guide_right", (0.022, 0.008, 0.052), (0, -GUIDE_Y, 0.026)),
            box_geom("camera_plateau", (0.00546, 0.042, 0.036),
                     (0.006855, 0.0523, 0.0546), friction=0.3),
            box_geom("bec", (0.055, 0.030, 0.014), (-0.034, 0, -0.015)),
            box_geom("control_bridge", (0.055, 0.025, 0.012), (0.024, 0, -0.014)),
        ]
    return geoms


def leg_names(x_index: int, side: int) -> tuple[str, float]:
    station = ("rear", "mid_rear", "mid_front", "front")[x_index]
    lateral = "left" if side > 0 else "right"
    return f"{lateral}_{station}", side * (140, 110, 70, 40)[x_index]


def leg_xml(x_index: int, side: int) -> list[str]:
    prefix, yaw_deg = leg_names(x_index, side)
    yaw = math.radians(yaw_deg)
    yaw_quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    pitch_quat = (math.cos(TIBIA_PITCH / 2), 0, math.sin(TIBIA_PITCH / 2), 0)
    coxa_mass, coxa_com, coxa_inertia = mass_properties(COXA_PARTS)
    tibia_mass, tibia_com, tibia_inertia = mass_properties(TIBIA_PARTS)
    x = HIP_X[x_index]
    y = side * HIP_Y

    return [
        f'<body name="{prefix}_coxa" pos="{fmt((x, y, HIP_Z))}" quat="{fmt(yaw_quat)}">',
        f'  <inertial pos="{fmt(coxa_com)}" mass="{fmt(coxa_mass)}" '
        f'diaginertia="{fmt(coxa_inertia)}"/>',
        f'  <joint name="{prefix}_hip" class="hip_joint" axis="0 0 1"/>',
        f'  <geom name="{prefix}_coxa_visual" class="visual" type="mesh" '
        f'mesh="coxa_visual" pos="0 0 -0.0035"/>',
        f'  <geom name="{prefix}_coxa_collision" class="footprint_collision" '
        f'type="capsule" size="0.007" fromto="0 0 0 {fmt(COXA_LENGTH)} 0 0"/>',
        visual_box(f"{prefix}_knee_servo_visual", (0.020, 0.029, 0.034),
                   (COXA_LENGTH, 0, 0), rgba="0.16 0.18 0.22 1",
                   quat="0.707106781 0 0 0.707106781"),
        f'  <geom name="{prefix}_knee_servo" class="collision" type="box" '
        f'size="0.01 0.0145 0.017" pos="{fmt((COXA_LENGTH, 0, 0))}" '
        f'quat="0.707106781 0 0 0.707106781"/>',
        f'  <body name="{prefix}_tibia" pos="{fmt((COXA_LENGTH, 0, 0))}" '
        f'quat="{fmt(pitch_quat)}">',
        f'    <inertial pos="{fmt(tibia_com)}" mass="{fmt(tibia_mass)}" '
        f'diaginertia="{fmt(tibia_inertia)}"/>',
        f'    <joint name="{prefix}_knee" class="knee_joint" axis="0 1 0"/>',
        f'    <geom name="{prefix}_tibia_visual" class="visual" type="mesh" '
        f'mesh="tibia_visual" pos="0 0 -0.0045"/>',
        f'    <geom name="{prefix}_foot_visual" class="visual" type="mesh" '
        f'mesh="foot_visual" pos="{fmt((TIBIA_LENGTH, 0, 0))}"/>',
        f'    <geom name="{prefix}_tibia_collision" class="footprint_collision" '
        f'type="capsule" size="0.0055" fromto="0.008 0 0 {fmt(TIBIA_LENGTH)} 0 0"/>',
        f'    <geom name="{prefix}_foot_collision" class="foot_collision" '
        f'type="box" size="{fmt(FOOT_HALF_SIZE)}" '
        f'pos="{fmt((TIBIA_LENGTH, 0, 0))}"/>',
        "  </body>",
        "</body>",
    ]


def generate(profile: str) -> str:
    root_mass, root_com, root_inertia = mass_properties(root_mass_parts())
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<!-- Generated by generate_model.py; collision profile: {profile}. -->',
        f'<mujoco model="arachne15_{profile}">',
        '  <compiler angle="radian" meshdir="../build/stl" autolimits="true"/>',
        '  <option timestep="0.002" gravity="0 0 -9.80665" integrator="implicitfast"/>',
        '  <default>',
        '    <default class="visual">',
        '      <geom contype="0" conaffinity="0" group="2" rgba="0.18 0.22 0.27 1"/>',
        '    </default>',
        '    <default class="collision">',
        '      <geom friction="0.55 0.005 0.0001" condim="3" rgba="0.3 0.65 0.8 0.25"/>',
        '    </default>',
        '    <default class="footprint_collision">',
        '      <geom friction="0.65 0.005 0.0001" condim="3" rgba="0.25 0.7 0.5 0.25"/>',
        '    </default>',
        '    <default class="foot_collision">',
        '      <geom friction="0.9 0.01 0.0002" condim="4" rgba="0.1 0.1 0.12 0.35"/>',
        '    </default>',
        '    <default class="hip_joint">',
        '      <joint type="hinge" range="-0.55 0.55" damping="0.02" armature="0.00005"/>',
        '    </default>',
        '    <default class="knee_joint">',
        '      <joint type="hinge" range="-0.70 0.90" damping="0.02" armature="0.00005"/>',
        '    </default>',
        '  </default>',
        '  <asset>',
        '    <mesh name="chassis_visual" file="chassis.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="phone_guide_visual" file="phone_guide.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="battery_cradle_visual" file="battery_cradle.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="coxa_visual" file="coxa_link.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="tibia_visual" file="tibia_link.stl" scale="0.001 0.001 0.001"/>',
        '    <mesh name="foot_visual" file="foot_pad.stl" scale="0.001 0.001 0.001"/>',
        '  </asset>',
        '  <worldbody>',
        '    <geom name="floor" type="plane" size="2 2 0.05" friction="0.9 0.01 0.0002"/>',
        f'    <body name="base" pos="0 0 {fmt(ROOT_WORLD_Z)}">',
        '      <freejoint/>',
        f'      <inertial pos="{fmt(root_com)}" mass="{fmt(root_mass)}" '
        f'diaginertia="{fmt(root_inertia)}"/>',
        '      <geom name="chassis_visual" class="visual" type="mesh" mesh="chassis_visual"/>',
        f'      <geom name="guide_left_visual" class="visual" type="mesh" '
        f'mesh="phone_guide_visual" pos="0 {fmt(GUIDE_Y)} 0"/>',
        f'      <geom name="guide_right_visual" class="visual" type="mesh" '
        f'mesh="phone_guide_visual" pos="0 {fmt(-GUIDE_Y)} 0" quat="0 0 0 1"/>',
        '      <geom name="battery_cradle_visual" class="visual" type="mesh" '
        'mesh="battery_cradle_visual" pos="0 0 -0.028"/>',
        f'      <geom name="iphone_visual" class="visual" type="box" '
        f'size="{fmt(tuple(v / 2 for v in PHONE_SIZE))}" pos="{fmt(PHONE_CENTER)}" '
        f'rgba="0.18 0.2 0.24 1"/>',
    ]
    lines += indent(root_visual_components(), 3)
    lines += indent(root_collision_geoms(profile), 3)
    for side in (-1, 1):
        for index in range(len(HIP_X)):
            lines += indent(leg_xml(index, side), 3)
    lines += [
        '    </body>',
        '  </worldbody>',
        '  <contact>',
    ]
    for side in (-1, 1):
        for index in range(len(HIP_X)):
            prefix, _ = leg_names(index, side)
            lines += [
                f'    <exclude body1="base" body2="{prefix}_coxa"/>',
                f'    <exclude body1="{prefix}_coxa" body2="{prefix}_tibia"/>',
            ]
    lines += [
        '  </contact>',
        '  <actuator>',
    ]
    for side in (-1, 1):
        for index in range(len(HIP_X)):
            prefix, _ = leg_names(index, side)
            lines += [
                f'    <motor name="{prefix}_hip_motor" joint="{prefix}_hip" '
                f'ctrlrange="-{fmt(DESIGN_TORQUE)} {fmt(DESIGN_TORQUE)}" ctrllimited="true"/>',
                f'    <motor name="{prefix}_knee_motor" joint="{prefix}_knee" '
                f'ctrlrange="-{fmt(DESIGN_TORQUE)} {fmt(DESIGN_TORQUE)}" ctrllimited="true"/>',
            ]
    lines += [
        '  </actuator>',
        '</mujoco>',
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail if checked-in XML differs from generated output")
    args = parser.parse_args()
    mismatches = []
    outputs: list[tuple[Path, str]] = []
    for profile in ("training", "validation"):
        output = generate(profile)
        outputs.append((ROOT / f"arachne15_{profile}.xml", output))
        outputs.append((
            BUNDLED_ASSET_ROOT / f"arachne15_{profile}.xml",
            output.replace('meshdir="../build/stl"', 'meshdir="meshes"'),
        ))
    for path, output in outputs:
        if args.check:
            if not path.exists() or path.read_text() != output:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output)
            print(path)
    source_meshes = ROOT.parent / "build/stl"
    bundled_meshes = BUNDLED_ASSET_ROOT / "meshes"
    for filename in VISUAL_MESH_FILES:
        source = source_meshes / filename
        destination = bundled_meshes / filename
        if args.check:
            if not destination.exists() or destination.read_bytes() != source.read_bytes():
                mismatches.append(str(destination))
        else:
            bundled_meshes.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            print(destination)
    if mismatches:
        raise SystemExit("stale generated assets: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
