#!/usr/bin/env python3
"""Dependency-free kinematic verification for Arachne Reveal.

This checks the exact compact joint targets shared with Swift. It measures the
complete articulated XY envelope (chassis, coxae, tibias, and feet), verifies
an eight-foot support polygon, and rejects inter-leg beam collisions. It is an
engineering pre-check; printed brackets and servo cables still require a slow
physical swept-volume test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIP_POSITIONS = (
    (-0.060, -0.092, 0.022), (-0.024, -0.092, 0.022),
    ( 0.024, -0.092, 0.022), ( 0.060, -0.092, 0.022),
    (-0.060,  0.092, 0.022), (-0.024,  0.092, 0.022),
    ( 0.024,  0.092, 0.022), ( 0.060,  0.092, 0.022),
)
FIXED_HIP_YAWS = tuple(math.radians(value) for value in (
    -140, -110, -70, -40, 140, 110, 70, 40))
COXA_LENGTH = 0.050
TIBIA_LENGTH = 0.105
FIXED_TIBIA_PITCH = math.radians(65)
HIP_LIMIT = 0.55
KNEE_LIMIT = (-0.70, 0.90)
COMPACT_HIPS = (-0.52, -0.52, 0.52, 0.52,
                 0.52,  0.52, -0.52, -0.52)
COMPACT_KNEE = 0.82
CHASSIS_HALF_EXTENT = (0.077, 0.106)
COXA_RADIUS = 0.007
TIBIA_RADIUS = 0.0055
ENVELOPE_MARGIN = 0.010


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, value):
    return tuple(x * value for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def leg_points(leg: int, hip_offset: float, knee_offset: float):
    hip = HIP_POSITIONS[leg]
    yaw = FIXED_HIP_YAWS[leg] + hip_offset
    direction = (math.cos(yaw), math.sin(yaw), 0.0)
    knee = add(hip, scale(direction, COXA_LENGTH))
    pitch = FIXED_TIBIA_PITCH + knee_offset
    foot = add(knee, (
        TIBIA_LENGTH * math.cos(pitch) * direction[0],
        TIBIA_LENGTH * math.cos(pitch) * direction[1],
        -TIBIA_LENGTH * math.sin(pitch),
    ))
    return hip, knee, foot


def envelope(hip_offsets, knee_offsets):
    points = [
        (-CHASSIS_HALF_EXTENT[0], -CHASSIS_HALF_EXTENT[1], 0),
        ( CHASSIS_HALF_EXTENT[0],  CHASSIS_HALF_EXTENT[1], 0),
    ]
    for leg in range(8):
        points.extend(leg_points(leg, hip_offsets[leg], knee_offsets[leg]))
    low = tuple(min(point[axis] for point in points) - ENVELOPE_MARGIN
                for axis in (0, 1))
    high = tuple(max(point[axis] for point in points) + ENVELOPE_MARGIN
                 for axis in (0, 1))
    dimensions = tuple(high[axis] - low[axis] for axis in (0, 1))
    return low, high, dimensions, dimensions[0] * dimensions[1]


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def origin_margin(hull):
    margins = []
    for index, point in enumerate(hull):
        following = hull[(index + 1) % len(hull)]
        dx = following[0] - point[0]
        dy = following[1] - point[1]
        margins.append((dx * -point[1] - dy * -point[0])
                       / math.hypot(dx, dy))
    return min(margins)


def segment_distance(p0, p1, q0, q1):
    # Closest points of two finite 3D segments, from the standard constrained
    # line-line solution with endpoint clamping.
    u, v, w = sub(p1, p0), sub(q1, q0), sub(p0, q0)
    a, b, c, d, e = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    denominator = a * c - b * b
    small = 1e-12
    if denominator < small:
        s_num, s_den, t_num, t_den = 0.0, 1.0, e, c
    else:
        s_num, s_den = b * e - c * d, denominator
        t_num, t_den = a * e - b * d, denominator
        if s_num < 0:
            s_num, t_num, t_den = 0.0, e, c
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, e + b, c
    if t_num < 0:
        t_num = 0.0
        if -d < 0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num, s_den = -d, a
    elif t_num > t_den:
        t_num = t_den
        if -d + b < 0:
            s_num = 0.0
        elif -d + b > a:
            s_num = s_den
        else:
            s_num, s_den = -d + b, a
    s = 0.0 if abs(s_num) < small else s_num / s_den
    t = 0.0 if abs(t_num) < small else t_num / t_den
    return norm(sub(add(w, scale(u, s)), scale(v, t)))


def compact_segment_clearance():
    segments = []
    for leg in range(8):
        hip, knee, foot = leg_points(leg, COMPACT_HIPS[leg], COMPACT_KNEE)
        tibia_start = add(knee, scale(sub(foot, knee), 0.008 / TIBIA_LENGTH))
        segments.extend((
            (leg, "coxa", hip, knee, COXA_RADIUS),
            (leg, "tibia", tibia_start, foot, TIBIA_RADIUS),
        ))
    minimum = math.inf
    pair = None
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1:]:
            if first[0] == second[0]:
                continue
            clearance = segment_distance(
                first[2], first[3], second[2], second[3]) - first[4] - second[4]
            if clearance < minimum:
                minimum, pair = clearance, (first[:2], second[:2])
    return minimum, pair


def report():
    deployed = envelope((0.0,) * 8, (0.0,) * 8)
    compact = envelope(COMPACT_HIPS, (COMPACT_KNEE,) * 8)
    feet = [leg_points(leg, COMPACT_HIPS[leg], COMPACT_KNEE)[2][:2]
            for leg in range(8)]
    support_margin = origin_margin(convex_hull(feet))
    clearance, closest_pair = compact_segment_clearance()
    area_ratio = compact[3] / deployed[3]
    checks = {
        "targets_inside_mechanical_limits": (
            max(abs(value) for value in COMPACT_HIPS) <= HIP_LIMIT
            and KNEE_LIMIT[0] <= COMPACT_KNEE <= KNEE_LIMIT[1]),
        "compact_area_ratio_at_most_0_65": area_ratio <= 0.65,
        "inter_leg_beam_clearance_at_least_8mm": clearance >= 0.008,
        "support_polygon_margin_at_least_65mm": support_margin >= 0.065,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "kinematic pre-check; servo housings, cables, and printed brackets require physical sweep validation",
        "compact_joint_targets_rad": [
            value for pair in zip(COMPACT_HIPS, (COMPACT_KNEE,) * 8)
            for value in pair
        ],
        "deployed_articulated_envelope_mm": [value * 1000 for value in deployed[2]],
        "compact_articulated_envelope_mm": [value * 1000 for value in compact[2]],
        "compact_to_deployed_area_ratio": area_ratio,
        "area_reduction_fraction": 1 - area_ratio,
        "compact_support_polygon_origin_margin_mm": support_margin * 1000,
        "minimum_inter_leg_beam_clearance_mm": clearance * 1000,
        "closest_beam_pair": closest_pair,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = report()
    print(json.dumps(result, indent=2))
    if not args.check:
        output = ROOT / "build/reveal_pose_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
