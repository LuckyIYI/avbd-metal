#!/usr/bin/env python3
"""Normalize a checked classic OBJ into the simulator's Z-up meter frame.

The conversion is deliberately smaller than a general mesh rewriter: it uses
the convex cooker's strict OBJ parser, bakes a uniform source-axis scale and a
right-handed source-up rotation, centers the horizontal bounds, and places the
lowest vertex on Z=0.  Output contains only positions and triangles so ModelIO
and the collision cooker consume exactly the same geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Optional, Sequence

from cook_convex_asset import (
    CookError,
    Mesh,
    atomic_write_many,
    baked_mesh,
    parse_obj,
    point_f32,
)


def normalized_mesh(
    mesh: Mesh, scale: float, up_axis: str
) -> tuple[Mesh, tuple[float, float, float]]:
    if not math.isfinite(scale) or scale <= 0:
        raise CookError("normalization scale must be positive")
    baked = baked_mesh(mesh, (scale, scale, scale), up_axis)
    lo = tuple(min(point[axis] for point in baked.vertices) for axis in range(3))
    hi = tuple(max(point[axis] for point in baked.vertices) for axis in range(3))
    offset = point_f32((
        -0.5 * (lo[0] + hi[0]),
        -0.5 * (lo[1] + hi[1]),
        -lo[2],
    ))
    vertices = tuple(point_f32((
        point[0] + offset[0],
        point[1] + offset[1],
        point[2] + offset[2],
    )) for point in baked.vertices)
    return Mesh(vertices, baked.triangles), offset


def encode_obj(
    mesh: Mesh,
    *,
    label: str,
    source_sha256: str,
    scale: float,
    up_axis: str,
    offset: tuple[float, float, float],
) -> bytes:
    lines = [
        f"# {label}",
        "# Deterministic AVBD Z-up meter normalization",
        f"# input-sha256 {source_sha256}",
        f"# source-up-axis {up_axis}",
        f"# uniform-scale {scale:.9g}",
        "# post-bake-translation " + " ".join(f"{value:.9g}" for value in offset),
        f"# vertices {len(mesh.vertices)} triangles {len(mesh.triangles)}",
    ]
    lines.extend(
        "v " + " ".join(f"{value:.9g}" for value in point)
        for point in mesh.vertices
    )
    lines.extend(
        "f " + " ".join(str(index + 1) for index in triangle)
        for triangle in mesh.triangles
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-up-axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--uniform-scale", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    try:
        source = args.input.read_bytes()
        parsed = parse_obj(source, str(args.input))
        mesh, offset = normalized_mesh(
            parsed, args.uniform_scale, args.source_up_axis
        )
        encoded = encode_obj(
            mesh,
            label=args.label,
            source_sha256=hashlib.sha256(source).hexdigest(),
            scale=args.uniform_scale,
            up_axis=args.source_up_axis,
            offset=offset,
        )
        atomic_write_many([(args.output, encoded)], protected_paths=(args.input,))
    except (CookError, OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
