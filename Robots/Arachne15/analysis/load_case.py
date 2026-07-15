#!/usr/bin/env python3
"""Conservative static/dynamic feasibility checks for Arachne-15.

This is an engineering pre-check, not a substitute for prototype load,
thermal, fatigue, tip-over, and gait testing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

G = 9.80665

# Source-controlled mass budget. The printed-part value rounds up the generated
# mesh report's full-solid PETG/TPU estimate, so ordinary infill cannot make the
# robot heavier than this line unless the CAD or material choices change.
MASS_G = {
    "iphone_15_pro": 187.0,
    "xc330_servos_16x": 16 * 23.0,
    "printed_chassis_dock_links_feet_full_solid_upper_bound": 275.0,
    "robotis_frames_horns_fasteners": 85.0,
    "2s_2200mah_battery": 135.0,
    "5v_20a_bec_and_distribution": 45.0,
    "wireless_ttl_bridge": 25.0,
    "wiring_padding_and_margin": 95.0,
}

# Conservative component-centre heights for the landscape camera-forward phone
# layout. These are used only for a first-order level-floor tip-over check.
MASS_Z_MM = {
    "iphone_15_pro": 118.3,
    "xc330_servos_16x": 90.0,
    "printed_chassis_dock_links_feet_full_solid_upper_bound": 90.0,
    "robotis_frames_horns_fasteners": 90.0,
    "2s_2200mah_battery": 50.0,
    "5v_20a_bec_and_distribution": 55.0,
    "wireless_ttl_bridge": 65.0,
    "wiring_padding_and_margin": 90.0,
}

STALL_TORQUE_NM = 0.93
ROBOTIS_GENERAL_USE_FRACTION = 0.20
TIBIA_LENGTH_M = 0.105
TIBIA_PITCH_DEG = 65.0
DYNAMIC_FACTOR = 1.50
MIN_STANCE_LEGS = 5
TARGET_MIN_SAFETY_FACTOR = 1.15

HIP_XS_M = [-0.069, -0.025, 0.025, 0.069]
HIP_Y_M = 0.063
COXA_M = 0.050


def outward_angle_deg(x: float, side: int) -> float:
    base = 140 if x < -0.044 else 110 if x < 0 else 70 if x < 0.044 else 40
    return side * base


def foot_positions() -> list[tuple[float, float]]:
    radial = COXA_M + TIBIA_LENGTH_M * math.cos(math.radians(TIBIA_PITCH_DEG))
    result = []
    for side in (-1, 1):
        for x in HIP_XS_M:
            a = math.radians(outward_angle_deg(x, side))
            result.append((x + radial * math.cos(a),
                           side * HIP_Y_M + radial * math.sin(a)))
    return result


def cross(o: tuple[float, float], a: tuple[float, float],
          b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def origin_margin(hull: list[tuple[float, float]]) -> float:
    """Minimum inward distance from origin to a CCW convex hull edge."""
    distances = []
    for i, p in enumerate(hull):
        q = hull[(i + 1) % len(hull)]
        dx, dy = q[0] - p[0], q[1] - p[1]
        distances.append((dx * (-p[1]) - dy * (-p[0])) / math.hypot(dx, dy))
    return min(distances)


def main() -> None:
    total_mass_kg = sum(MASS_G.values()) / 1000.0
    usable_torque = STALL_TORQUE_NM * ROBOTIS_GENERAL_USE_FRACTION
    horizontal_arm = TIBIA_LENGTH_M * math.cos(math.radians(TIBIA_PITCH_DEG))

    stance_cases = {}
    for legs in (4, 5, 6, 7):
        per_leg_force = total_mass_kg * G / legs
        required = per_leg_force * horizontal_arm * DYNAMIC_FACTOR
        stance_cases[str(legs)] = {
            "per_leg_vertical_force_n": per_leg_force,
            "dynamic_knee_torque_nm": required,
            "safety_factor_at_20pct_stall": usable_torque / required,
            "passes": usable_torque / required >= TARGET_MIN_SAFETY_FACTOR,
        }

    feet = foot_positions()
    single_swing_margins = []
    for swing in range(len(feet)):
        hull = convex_hull([p for i, p in enumerate(feet) if i != swing])
        single_swing_margins.append(origin_margin(hull))

    estimated_com_height_mm = sum(
        MASS_G[name] * MASS_Z_MM[name] for name in MASS_G
    ) / sum(MASS_G.values())
    worst_support_margin_mm = min(single_swing_margins) * 1000
    lateral_accel_to_tip_g = worst_support_margin_mm / estimated_com_height_mm

    worst_case = stance_cases[str(MIN_STANCE_LEGS)]
    report = {
        "status": "PASS" if worst_case["passes"] and min(single_swing_margins) > 0
        else "FAIL",
        "scope": "analytical pre-check; flat floor, slow wave gait, no prototype evidence",
        "mass_budget_g": MASS_G,
        "total_design_mass_g": total_mass_kg * 1000,
        "actuator": {
            "model": "ROBOTIS XC330-M288-T",
            "stall_torque_nm_at_5v": STALL_TORQUE_NM,
            "design_fraction_of_stall": ROBOTIS_GENERAL_USE_FRACTION,
            "design_torque_nm": usable_torque,
        },
        "geometry": {
            "tibia_length_mm": TIBIA_LENGTH_M * 1000,
            "tibia_pitch_deg_below_horizontal": TIBIA_PITCH_DEG,
            "horizontal_knee_moment_arm_mm": horizontal_arm * 1000,
            "nominal_foot_positions_m": feet,
            "minimum_dock_to_inner_hip_clearance_mm": 3.0,
        },
        "load_cases": stance_cases,
        "required_gait": {
            "minimum_stance_legs": MIN_STANCE_LEGS,
            "recommended": "one-leg-at-a-time wave gait (7 stance legs)",
            "dynamic_factor": DYNAMIC_FACTOR,
            "minimum_safety_factor": TARGET_MIN_SAFETY_FACTOR,
        },
        "stability": {
            "single_swing_min_origin_margin_mm": min(single_swing_margins) * 1000,
            "single_swing_max_origin_margin_mm": max(single_swing_margins) * 1000,
            "passes_origin_inside_support_polygon": min(single_swing_margins) > 0,
            "estimated_center_of_mass_height_mm": estimated_com_height_mm,
            "estimated_lateral_acceleration_to_tip_g": lateral_accel_to_tip_g,
            "note": "first-order rigid-body estimate; validate on a tethered prototype",
        },
        "power": {
            "servo_stall_current_each_a": 1.80,
            "all_servo_stall_current_a": 16 * 1.80,
            "recommended_bec": "regulated 5 V, >=20 A continuous, >=30 A peak",
            "required_controls": [
                "four separately fused power-injection branches",
                "firmware current limits and acceleration profiles",
                "physical master power switch and 25-30 A main fuse",
                "never source actuator power from the iPhone USB-C port",
            ],
        },
    }

    out = Path(__file__).resolve().parents[1] / "build" / "load_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
