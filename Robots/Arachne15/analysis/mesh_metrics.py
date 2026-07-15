#!/usr/bin/env python3
"""Dependency-free STL bounds, volume, and print-mass verification."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path


# One exported file per unique part.  Quantities describe one complete robot.
# Densities are intentionally toward the high end of common filament data.
ASSEMBLY = {
    "battery_cradle.stl": {"quantity": 1, "material": "PETG", "density_g_cm3": 1.27},
    "chassis.stl": {"quantity": 1, "material": "PETG", "density_g_cm3": 1.27},
    "coxa_link.stl": {"quantity": 8, "material": "PETG", "density_g_cm3": 1.27},
    "foot_pad.stl": {"quantity": 8, "material": "TPU 95A", "density_g_cm3": 1.21},
    "phone_tray.stl": {"quantity": 1, "material": "PETG", "density_g_cm3": 1.27},
    "retainer_clip.stl": {"quantity": 4, "material": "PETG", "density_g_cm3": 1.27},
    "tibia_link.stl": {"quantity": 8, "material": "PETG", "density_g_cm3": 1.27},
}


def triangles(path: Path):
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            for i in range(count):
                off = 84 + i * 50 + 12
                yield tuple(struct.unpack_from("<9f", data, off))
            return
    text = data.decode("utf-8", errors="strict")
    vertices = []
    for line in text.splitlines():
        words = line.strip().split()
        if len(words) == 4 and words[0] == "vertex":
            vertices.append(tuple(map(float, words[1:])))
            if len(vertices) == 3:
                yield vertices[0] + vertices[1] + vertices[2]
                vertices.clear()


def metrics(path: Path) -> dict:
    count = 0
    signed_six_volume = 0.0
    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    for tri in triangles(path):
        count += 1
        a, b, c = tri[0:3], tri[3:6], tri[6:9]
        signed_six_volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        for vertex in (a, b, c):
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])
    if count == 0:
        raise ValueError(f"{path}: no STL triangles")
    return {
        "file": str(path),
        "triangles": count,
        "bounds_min_mm": mins,
        "bounds_max_mm": maxs,
        "size_mm": [maxs[i] - mins[i] for i in range(3)],
        "solid_volume_cm3": abs(signed_six_volume) / 6.0 / 1000.0,
    }


def main() -> None:
    audit_only = len(sys.argv) > 1 and sys.argv[1] == "--audit-only"
    paths = [Path(arg) for arg in sys.argv[2 if audit_only else 1:]]
    if not paths:
        raise SystemExit("usage: mesh_metrics.py [--audit-only] build/stl/*.stl")
    parts = [metrics(path) for path in paths]
    if audit_only:
        print(json.dumps({"status": "PASS", "parts": parts}, indent=2))
        return
    unknown = sorted({Path(part["file"]).name for part in parts} - ASSEMBLY.keys())
    missing = sorted(ASSEMBLY.keys() - {Path(part["file"]).name for part in parts})
    if unknown or missing:
        raise SystemExit(f"assembly manifest mismatch: unknown={unknown}, missing={missing}")

    full_solid_mass_g = 0.0
    for part in parts:
        spec = ASSEMBLY[Path(part["file"]).name]
        part.update(spec)
        part["assembly_full_solid_mass_g"] = (
            part["solid_volume_cm3"] * spec["quantity"] * spec["density_g_cm3"]
        )
        full_solid_mass_g += part["assembly_full_solid_mass_g"]

    report = {
        "status": "PASS",
        "method": "closed-mesh signed volume times conservative nominal density",
        "parts": parts,
        "assembly": {
            "unique_printed_parts": len(parts),
            "total_printed_pieces": sum(spec["quantity"] for spec in ASSEMBLY.values()),
            "full_solid_mass_g": full_solid_mass_g,
            "load_budget_rounded_up_g": 275.0,
            "passes_load_budget": full_solid_mass_g <= 275.0,
        },
    }
    if not report["assembly"]["passes_load_budget"]:
        report["status"] = "FAIL"
    output = paths[0].resolve().parents[1] / "mesh_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
