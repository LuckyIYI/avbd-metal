#!/usr/bin/env python3
"""Generate AVBD's bounded Unitree H1 collision support sets.

The generator is deliberately offline and dependency-free.  It accepts the
three commit-pinned binary STL files as local inputs, verifies their SHA-256
digests, and emits both the Swift payload and its machine-readable provenance
manifest.  No clean build or test needs the source meshes or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REVISION = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
RAW_ROOT = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{REVISION}/unitree_h1/assets"
)
PROBE_AXIS_LIMIT = 8
VERTEX_LIMIT = 64
GENERATOR_VERSION = 1


@dataclass(frozen=True)
class MeshSpec:
    swift_name: str
    file_name: str
    sha256: str


MESHES = (
    MeshSpec(
        "leftAnkle",
        "left_ankle_link.stl",
        "f623223b59d0f8cd86a799f409f4e8680da3e48d9b60e9f4bf9dad5c5dc8ac25",
    ),
    MeshSpec(
        "rightAnkle",
        "right_ankle_link.stl",
        "a72e7b10d3961b5c5bb7b0ca09f20cbc203619f5ded33ca030517c06298d2fd9",
    ),
    MeshSpec(
        "torso",
        "torso_link.stl",
        "3ab785cbfbd851b0e9c59a46871aabc4718f3f8e726e21a2d14360d217ec7ba8",
    ),
)

Point = tuple[float, float, float]
Direction = tuple[int, int, int]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_binary_stl(path: Path, expected_sha256: str) -> tuple[list[Point], int, int]:
    data = path.read_bytes()
    digest = sha256(data)
    if digest != expected_sha256:
        raise ValueError(
            f"{path}: SHA-256 {digest} does not match pinned {expected_sha256}"
        )
    if len(data) < 84:
        raise ValueError(f"{path}: truncated binary STL")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if len(data) != expected_size:
        raise ValueError(
            f"{path}: binary STL size is {len(data)}, expected {expected_size}"
        )

    vertices: set[Point] = set()
    for triangle in range(triangle_count):
        vertex_offset = 84 + triangle * 50 + 12
        for corner in range(3):
            point = struct.unpack_from("<3f", data, vertex_offset + 12 * corner)
            if not all(math.isfinite(component) for component in point):
                raise ValueError(f"{path}: non-finite vertex in triangle {triangle}")
            vertices.add(point)
    if len(vertices) < 4:
        raise ValueError(f"{path}: fewer than four unique vertices")
    return sorted(vertices), triangle_count, len(data)


def gcd3(x: int, y: int, z: int) -> int:
    return math.gcd(math.gcd(abs(x), abs(y)), abs(z))


def probe_directions(axis_limit: int) -> list[Direction]:
    """Return a deterministic, antipodally symmetric primitive lattice."""
    directions = []
    for x in range(-axis_limit, axis_limit + 1):
        for y in range(-axis_limit, axis_limit + 1):
            for z in range(-axis_limit, axis_limit + 1):
                if (x or y or z) and gcd3(x, y, z) == 1:
                    directions.append((x, y, z))
    return directions


def dot(point: Point, direction: Direction) -> float:
    return (
        point[0] * direction[0]
        + point[1] * direction[1]
        + point[2] * direction[2]
    )


def aabb(points: Sequence[Point]) -> tuple[Point, Point]:
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def normalized_points(points: Sequence[Point]) -> list[Point]:
    lo, hi = aabb(points)
    center = tuple((lo[axis] + hi[axis]) * 0.5 for axis in range(3))
    half_extent = tuple((hi[axis] - lo[axis]) * 0.5 for axis in range(3))
    if any(extent <= 0 for extent in half_extent):
        raise ValueError("source mesh has a degenerate AABB")
    return [
        tuple(
            (point[axis] - center[axis]) / half_extent[axis]
            for axis in range(3)
        )
        for point in points
    ]


def support_targets(
    points: Sequence[Point], directions: Sequence[Direction]
) -> tuple[list[float], list[int]]:
    values: list[float] = []
    indices: list[int] = []
    for direction in directions:
        best_index = 0
        best_value = dot(points[0], direction)
        for index in range(1, len(points)):
            value = dot(points[index], direction)
            # `points` is lexicographically sorted. Keeping the first exact
            # tie makes planar support selection independent of STL ordering.
            if value > best_value:
                best_index = index
                best_value = value
        values.append(best_value)
        indices.append(best_index)
    return values, indices


def reduce_support_set(
    points: Sequence[Point], directions: Sequence[Direction], vertex_limit: int
) -> list[Point]:
    """Greedily minimize normalized support-function error at a hard cap."""
    normalized = normalized_points(points)
    targets, target_indices = support_targets(normalized, directions)
    direction_indices = {direction: i for i, direction in enumerate(directions)}
    axis_seeds = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    current = [-math.inf] * len(directions)
    selected: list[int] = []
    selected_set: set[int] = set()
    for iteration in range(vertex_limit):
        if iteration < len(axis_seeds):
            direction_index = direction_indices[axis_seeds[iteration]]
        else:
            direction_index = -1
            greatest_error_metric = -1.0
            for index, direction in enumerate(directions):
                error = targets[index] - current[index]
                norm_squared = (
                    direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2]
                )
                error_metric = error * error / norm_squared
                if error_metric > greatest_error_metric:
                    direction_index = index
                    greatest_error_metric = error_metric

        point_index = target_indices[direction_index]
        if point_index in selected_set:
            raise RuntimeError(
                "support refinement selected an existing point before reaching "
                f"the {vertex_limit}-vertex cap"
            )
        selected.append(point_index)
        selected_set.add(point_index)
        selected_point = normalized[point_index]
        for index, direction in enumerate(directions):
            current[index] = max(current[index], dot(selected_point, direction))

    # Runtime support mapping is insensitive to ordering. Sorting makes the
    # checked-in Swift diff stable even if the refinement is later refactored.
    return sorted(points[index] for index in selected)


def support_error(
    source: Sequence[Point], reduced: Sequence[Point], directions: Sequence[Direction]
) -> tuple[float, float]:
    squared_error_sum = 0.0
    maximum_error = 0.0
    for direction in directions:
        length = math.sqrt(
            direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2]
        )
        source_support = max(dot(point, direction) for point in source) / length
        reduced_support = max(dot(point, direction) for point in reduced) / length
        error = source_support - reduced_support
        # Every reduced point is an exact source point, so only round-off can
        # produce a tiny negative error.
        if error < -1e-12:
            raise RuntimeError("reduced support exceeds source support")
        error = max(0.0, error)
        maximum_error = max(maximum_error, error)
        squared_error_sum += error * error
    return maximum_error, math.sqrt(squared_error_sum / len(directions))


def mirrored_support_error(
    left: Sequence[Point], right: Sequence[Point], directions: Sequence[Direction]
) -> tuple[float, float]:
    squared_error_sum = 0.0
    maximum_error = 0.0
    for direction in directions:
        reflected = (direction[0], -direction[1], direction[2])
        length = math.sqrt(
            direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2]
        )
        left_support = max(dot(point, direction) for point in left) / length
        right_support = max(dot(point, reflected) for point in right) / length
        error = abs(left_support - right_support)
        maximum_error = max(maximum_error, error)
        squared_error_sum += error * error
    return maximum_error, math.sqrt(squared_error_sum / len(directions))


def float_literal(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".9g")


def swift_source(hulls: dict[str, list[Point]]) -> bytes:
    lines = [
        "// Generated by Tools/generate_unitree_h1_collision_hulls.py; do not edit.",
        f"// MuJoCo Menagerie revision: {REVISION}",
        "// Inputs and reduction metrics: Assets/unitree_h1/COLLISION_HULLS_PROVENANCE.json",
        "",
        "import simd",
        "import SimCore",
        "",
        "/// Bounded support-map approximations generated directly from the pinned",
        "/// BSD-3-Clause Unitree H1 meshes in MuJoCo Menagerie. Every entry is an",
        "/// exact source-mesh vertex; deterministic farthest-error refinement fills",
        "/// the Metal narrow phase's 64-vertex budget. Coordinates are link-local.",
        "package enum UnitreeH1CollisionHulls {",
    ]
    for spec in MESHES:
        lines.append(f"    package static let {spec.swift_name}: [F3] = [")
        for point in hulls[spec.swift_name]:
            components = ", ".join(float_literal(value) for value in point)
            lines.append(f"        F3({components}),")
        lines.extend(("    ]", ""))
    lines.extend(("}", ""))
    return "\n".join(lines).encode("utf-8")


def packed_vertex_sha256(points: Iterable[Point]) -> str:
    payload = b"".join(struct.pack("<3f", *point) for point in points)
    return sha256(payload)


def manifest(
    generator_path: Path,
    license_path: Path,
    swift_path: Path,
    swift_data: bytes,
    records: dict[str, dict[str, object]],
    mirror_error: tuple[float, float],
    direction_count: int,
) -> bytes:
    meshes = []
    for spec in MESHES:
        record = records[spec.swift_name]
        meshes.append(
            {
                "name": spec.swift_name,
                "sourceFile": spec.file_name,
                "sourceURL": f"{RAW_ROOT}/{spec.file_name}",
                "sourceSHA256": spec.sha256,
                "sourceByteCount": record["sourceByteCount"],
                "sourceTriangleCount": record["sourceTriangleCount"],
                "sourceUniqueVertexCount": record["sourceUniqueVertexCount"],
                "sourceAABB": record["sourceAABB"],
                "generatedVertexCount": record["generatedVertexCount"],
                "generatedVerticesSHA256": record["generatedVerticesSHA256"],
                "generatedAABB": record["generatedAABB"],
                "symmetricSupportErrorMaxMeters": record["supportErrorMaxMeters"],
                "symmetricSupportErrorRMSMeters": record["supportErrorRMSMeters"],
            }
        )
    payload = {
        "schemaVersion": 1,
        "source": {
            "project": "MuJoCo Menagerie / Unitree H1",
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "BSD-3-Clause",
            "bundledLicenseFile": "LICENSE",
            "bundledLicenseSHA256": sha256(license_path.read_bytes()),
        },
        "generator": {
            "path": "Tools/generate_unitree_h1_collision_hulls.py",
            "sha256": sha256(generator_path.read_bytes()),
            "version": GENERATOR_VERSION,
            "runtime": "Python 3.9+ standard library only",
            "networkAccess": False,
            "input": "local commit-pinned binary STL files; SHA-256 is mandatory",
            "algorithm": (
                "Seed six AABB extrema, then repeatedly add the source support "
                "vertex at the greatest per-axis-AABB-normalized support error"
            ),
            "tieBreak": "lexicographically sorted source vertex, then probe order",
            "probeDirections": {
                "construction": (
                    "all primitive integer triples in [-8, 8]^3 except (0, 0, 0)"
                ),
                "axisLimit": PROBE_AXIS_LIMIT,
                "count": direction_count,
                "antipodallySymmetric": True,
            },
            "vertexLimit": VERTEX_LIMIT,
            "vertexLimitRationale": (
                "AVBD's Metal convex support path has a reviewed hard cap of 64 "
                "vertices per collider"
            ),
            "errorMeasurement": (
                "Physical-metre support-function error over every antipodal probe "
                "direction; max and RMS compare the full STL vertex set with the "
                "generated subset"
            ),
        },
        "output": {
            "swiftSource": str(swift_path),
            "swiftSourceSHA256": sha256(swift_data),
            "mirroredAnkleSupportDifferenceMaxMeters": mirror_error[0],
            "mirroredAnkleSupportDifferenceRMSMeters": mirror_error[1],
            "meshes": meshes,
        },
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_checked_in_outputs(repository_root: Path, manifest_path: Path) -> None:
    """Bind the generated payload to provenance without needing source STLs."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != 1:
        raise ValueError("unsupported collision-hull provenance schema")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("provenance source record is missing")
    if source.get("revision") != REVISION or source.get("license") != "BSD-3-Clause":
        raise ValueError("provenance source revision/license does not match generator")
    license_name = source.get("bundledLicenseFile")
    if not isinstance(license_name, str):
        raise ValueError("bundled license path is missing")
    license_path = manifest_path.parent / license_name
    if not license_path.is_file():
        raise ValueError(f"bundled license is missing: {license_path}")
    if sha256(license_path.read_bytes()) != source.get("bundledLicenseSHA256"):
        raise ValueError("bundled license SHA-256 does not match provenance")

    generator = payload.get("generator")
    if not isinstance(generator, dict) or generator.get("version") != GENERATOR_VERSION:
        raise ValueError("generator provenance is missing or unsupported")
    if generator.get("vertexLimit") != VERTEX_LIMIT:
        raise ValueError("manifest vertex limit does not match Metal contract")
    probes = generator.get("probeDirections")
    expected_direction_count = len(probe_directions(PROBE_AXIS_LIMIT))
    if (
        not isinstance(probes, dict)
        or probes.get("axisLimit") != PROBE_AXIS_LIMIT
        or probes.get("count") != expected_direction_count
        or probes.get("antipodallySymmetric") is not True
    ):
        raise ValueError("manifest support-probe contract is stale")
    generator_relative_path = generator.get("path")
    if not isinstance(generator_relative_path, str):
        raise ValueError("generator path is missing")
    generator_path = repository_root / generator_relative_path
    if generator_path.resolve() != Path(__file__).resolve():
        raise ValueError("manifest names a different generator")
    if sha256(generator_path.read_bytes()) != generator.get("sha256"):
        raise ValueError("generator SHA-256 does not match provenance")

    output = payload.get("output")
    if not isinstance(output, dict):
        raise ValueError("provenance output record is missing")
    swift_relative_path = output.get("swiftSource")
    if not isinstance(swift_relative_path, str):
        raise ValueError("generated Swift source path is missing")
    swift_path = repository_root / swift_relative_path
    swift_data = swift_path.read_bytes()
    if sha256(swift_data) != output.get("swiftSourceSHA256"):
        raise ValueError("generated Swift source SHA-256 does not match provenance")
    swift_text = swift_data.decode("utf-8")

    manifest_meshes = output.get("meshes")
    if not isinstance(manifest_meshes, list) or len(manifest_meshes) != len(MESHES):
        raise ValueError("manifest mesh records are missing")
    records = {
        record.get("name"): record
        for record in manifest_meshes
        if isinstance(record, dict)
    }
    for spec in MESHES:
        record = records.get(spec.swift_name)
        if not isinstance(record, dict):
            raise ValueError(f"manifest record is missing for {spec.swift_name}")
        if (
            record.get("sourceFile") != spec.file_name
            or record.get("sourceSHA256") != spec.sha256
            or record.get("sourceURL") != f"{RAW_ROOT}/{spec.file_name}"
        ):
            raise ValueError(f"source provenance is stale for {spec.swift_name}")
        if record.get("generatedVertexCount") != VERTEX_LIMIT:
            raise ValueError(f"invalid generated vertex count for {spec.swift_name}")
        maximum_error = record.get("symmetricSupportErrorMaxMeters")
        rms_error = record.get("symmetricSupportErrorRMSMeters")
        if (
            not isinstance(maximum_error, (int, float))
            or not isinstance(rms_error, (int, float))
            or not math.isfinite(maximum_error)
            or not math.isfinite(rms_error)
            or maximum_error < 0
            or rms_error < 0
            or rms_error > maximum_error
        ):
            raise ValueError(f"invalid support error metrics for {spec.swift_name}")

        block = re.search(
            rf"static let {re.escape(spec.swift_name)}: \[F3\] = \[(.*?)\n    \]",
            swift_text,
            re.DOTALL,
        )
        if block is None:
            raise ValueError(f"generated Swift array is missing for {spec.swift_name}")
        points = []
        for match in re.findall(r"F3\(([^)]+)\)", block.group(1)):
            components = tuple(float(value.strip()) for value in match.split(","))
            if len(components) != 3 or not all(math.isfinite(value) for value in components):
                raise ValueError(f"invalid Swift vertex for {spec.swift_name}")
            # Swift stores F3 as Float32. Repack/unpack before hashing so this
            # verifies the compiled payload, not Python's Float64 parse.
            points.append(struct.unpack("<3f", struct.pack("<3f", *components)))
        if len(points) != VERTEX_LIMIT or len(set(points)) != VERTEX_LIMIT:
            raise ValueError(f"generated Swift vertex budget is invalid for {spec.swift_name}")
        if packed_vertex_sha256(points) != record.get("generatedVerticesSHA256"):
            raise ValueError(f"generated vertex SHA-256 is stale for {spec.swift_name}")

    print(
        f"verified checked-in H1 collision hull provenance, generator, license, "
        f"and {len(MESHES) * VERTEX_LIMIT} Swift vertices"
    )


def write_or_check(path: Path, data: bytes, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != data:
            raise RuntimeError(f"generated output is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="directory containing the three pinned Menagerie binary STLs",
    )
    parser.add_argument(
        "--swift-output",
        type=Path,
        default=repository_root
        / "Sources/Robotics/UnitreeH1CollisionHulls.swift",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=repository_root
        / "Sources/Robotics/Assets/unitree_h1/COLLISION_HULLS_PROVENANCE.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in outputs differ; never write files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "verify checked-in generator/license/Swift hashes and payload shape "
            "without source STLs or network access"
        ),
    )
    arguments = parser.parse_args()

    if arguments.verify:
        verify_checked_in_outputs(repository_root, arguments.manifest_output)
        if arguments.source_dir is None:
            return 0
    if arguments.source_dir is None:
        parser.error("--source-dir is required unless --verify is used")

    directions = probe_directions(PROBE_AXIS_LIMIT)
    hulls: dict[str, list[Point]] = {}
    records: dict[str, dict[str, object]] = {}
    for spec in MESHES:
        source_path = arguments.source_dir / spec.file_name
        points, triangles, byte_count = parse_binary_stl(source_path, spec.sha256)
        reduced = reduce_support_set(points, directions, VERTEX_LIMIT)
        error_max, error_rms = support_error(points, reduced, directions)
        source_lo, source_hi = aabb(points)
        reduced_lo, reduced_hi = aabb(reduced)
        hulls[spec.swift_name] = reduced
        records[spec.swift_name] = {
            "sourceByteCount": byte_count,
            "sourceTriangleCount": triangles,
            "sourceUniqueVertexCount": len(points),
            "sourceAABB": {"min": source_lo, "max": source_hi},
            "generatedVertexCount": len(reduced),
            "generatedVerticesSHA256": packed_vertex_sha256(reduced),
            "generatedAABB": {"min": reduced_lo, "max": reduced_hi},
            "supportErrorMaxMeters": error_max,
            "supportErrorRMSMeters": error_rms,
        }

    mirror_error = mirrored_support_error(
        hulls["leftAnkle"], hulls["rightAnkle"], directions
    )
    swift_path = arguments.swift_output
    swift_data = swift_source(hulls)
    manifest_data = manifest(
        Path(__file__).resolve(),
        repository_root / "Sources/Robotics/Assets/unitree_h1/LICENSE",
        swift_path.relative_to(repository_root),
        swift_data,
        records,
        mirror_error,
        len(directions),
    )
    write_or_check(swift_path, swift_data, arguments.check)
    write_or_check(arguments.manifest_output, manifest_data, arguments.check)
    action = "verified" if arguments.check else "generated"
    print(
        f"{action} {swift_path} and {arguments.manifest_output} "
        f"from {len(directions)} symmetric support probes"
    )
    for spec in MESHES:
        record = records[spec.swift_name]
        print(
            f"  {spec.swift_name}: {record['generatedVertexCount']} vertices, "
            f"max {record['supportErrorMaxMeters']:.9g} m, "
            f"RMS {record['supportErrorRMSMeters']:.9g} m"
        )
    print(
        f"  mirrored ankles: max {mirror_error[0]:.9g} m, "
        f"RMS {mirror_error[1]:.9g} m"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
