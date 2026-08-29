#!/usr/bin/env python3
"""Cook deterministic, validated convex-compound assets for AVBD Metal.

The runtime consumes only the checked-in JSON produced by this tool.  CoACD is
an optional *offline* dependency: the default ``coacd`` method fails closed
when it is unavailable, while the explicit ``hull`` method uses the small
dependency-free deterministic hull implementation below.  No build or test of
the Swift package needs CoACD or NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import numbers
import os
from pathlib import Path
import struct
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = 1
ASSET_KIND = "avbd.convex-compound"
BUILTIN_HULL_BACKEND = "avbd-incremental-hull"
BUILTIN_HULL_VERSION = "1"
DEFAULT_MAX_VERTICES_PER_HULL = 64
MAX_VERTICES_PER_HULL = 64
MAX_TRIANGLES_PER_HULL = 124  # 2V - 4 for a closed triangular convex polytope.
MAX_EDGES_PER_HULL = 186  # 3V - 6 for the same topology.
# Clipping one face by another may need the sum of both boundary sizes in the
# Metal kernel's fixed 32-vertex workspace.
MAX_FACE_VERTICES = 16
MAX_HULLS = 256
MAX_ASSET_BYTES = 16 * 1024 * 1024
MAX_SOURCE_URI_BYTES = 4096
PINNED_COACD_VERSION = "1.0.11"
COOKER_CONTRACT_VERSION = 1
FLOAT32_EPSILON = 2.0 ** -23
FLOAT32_MIN_NORMAL = 2.0 ** -126

Point = tuple[float, float, float]
Triangle = tuple[int, int, int]


class CookError(ValueError):
    """An input or generated asset violated the cooker contract."""


@dataclass(frozen=True)
class Mesh:
    vertices: tuple[Point, ...]
    triangles: tuple[Triangle, ...]


@dataclass(frozen=True)
class CookParameters:
    threshold_meters: float = 0.05
    max_hulls: int = 64
    max_vertices_per_hull: int = DEFAULT_MAX_VERTICES_PER_HULL
    seed: int = 0
    weld_tolerance_relative: float = 1.0e-7
    weld_tolerance_absolute: float = 1.0e-9
    split_connected_components: bool = True
    coacd_mcts_nodes: int = 20
    coacd_mcts_iterations: int = 150
    coacd_mcts_max_depth: int = 3
    coacd_preprocess_mode: str = "auto"
    coacd_preprocess_resolution: int = 50
    coacd_resolution: int = 2_000
    coacd_merge: bool = True
    coacd_decimate: bool = True
    coacd_extrude: bool = False
    coacd_extrude_margin: float = 0.01
    coacd_pca: bool = False
    coacd_approximation_mode: str = "ch"

    def validate(self) -> None:
        for name, value in (
            ("threshold", self.threshold_meters),
            ("relative weld tolerance", self.weld_tolerance_relative),
            ("absolute weld tolerance", self.weld_tolerance_absolute),
            ("CoACD extrude margin", self.coacd_extrude_margin),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise CookError(f"{name} must be a real number, not a boolean")
        for name, value in (
            ("max hulls", self.max_hulls),
            ("max vertices per hull", self.max_vertices_per_hull),
            ("seed", self.seed),
            ("MCTS nodes", self.coacd_mcts_nodes),
            ("MCTS iterations", self.coacd_mcts_iterations),
            ("MCTS max depth", self.coacd_mcts_max_depth),
            ("preprocess resolution", self.coacd_preprocess_resolution),
            ("sampling resolution", self.coacd_resolution),
        ):
            if type(value) is not int:
                raise CookError(f"{name} must be an integer, not a boolean")
        for name, value in (
            ("connected-component splitting", self.split_connected_components),
            ("CoACD merge", self.coacd_merge),
            ("CoACD decimate", self.coacd_decimate),
            ("CoACD extrude", self.coacd_extrude),
            ("CoACD PCA", self.coacd_pca),
        ):
            if type(value) is not bool:
                raise CookError(f"{name} must be a boolean")
        if type(self.coacd_preprocess_mode) is not str:
            raise CookError("CoACD preprocess mode must be a string")
        if type(self.coacd_approximation_mode) is not str:
            raise CookError("CoACD approximation mode must be a string")

        finite_positive = (
            ("threshold", self.threshold_meters),
            ("relative weld tolerance", self.weld_tolerance_relative),
            ("absolute weld tolerance", self.weld_tolerance_absolute),
            ("CoACD extrude margin", self.coacd_extrude_margin),
        )
        for name, value in finite_positive:
            if not math.isfinite(value) or value <= 0:
                raise CookError(f"{name} must be finite and positive")
        if not 1 <= self.max_hulls <= MAX_HULLS:
            raise CookError(f"max hulls must be in 1...{MAX_HULLS}")
        if not 4 <= self.max_vertices_per_hull <= MAX_VERTICES_PER_HULL:
            raise CookError(
                f"max vertices per hull must be in 4...{MAX_VERTICES_PER_HULL}"
            )
        if not 0 <= self.seed <= 0x7FFF_FFFF:
            raise CookError("seed must be in 0...2147483647")
        for name, value in (
            ("MCTS nodes", self.coacd_mcts_nodes),
            ("MCTS iterations", self.coacd_mcts_iterations),
            ("MCTS max depth", self.coacd_mcts_max_depth),
            ("preprocess resolution", self.coacd_preprocess_resolution),
            ("sampling resolution", self.coacd_resolution),
        ):
            if not 0 < value <= 0xFFFF_FFFF:
                raise CookError(f"CoACD {name} must be positive")
        if self.coacd_preprocess_mode not in {"auto", "on", "off"}:
            raise CookError("CoACD preprocess mode must be auto, on, or off")
        if self.coacd_approximation_mode not in {"ch", "box"}:
            raise CookError("CoACD approximation mode must be ch or box")
        if not self.split_connected_components:
            raise CookError("connected-component splitting is required")

    def as_json(self) -> dict[str, Any]:
        return {
            "coacdDecimate": self.coacd_decimate,
            "coacdExtrude": self.coacd_extrude,
            "coacdExtrudeMargin": f32(self.coacd_extrude_margin),
            "coacdMCTSIterations": self.coacd_mcts_iterations,
            "coacdMCTSMaxDepth": self.coacd_mcts_max_depth,
            "coacdMCTSNodes": self.coacd_mcts_nodes,
            "coacdMerge": self.coacd_merge,
            "coacdPCA": self.coacd_pca,
            "coacdApproximationMode": self.coacd_approximation_mode,
            "coacdPreprocessMode": self.coacd_preprocess_mode,
            "coacdPreprocessResolution": self.coacd_preprocess_resolution,
            "coacdResolution": self.coacd_resolution,
            "coacdRealMetric": True,
            "maxHulls": self.max_hulls,
            "maxVerticesPerHull": self.max_vertices_per_hull,
            "seed": self.seed,
            "splitConnectedComponents": self.split_connected_components,
            "thresholdMeters": f32(self.threshold_meters),
            "weldToleranceAbsolute": f32(self.weld_tolerance_absolute),
            "weldToleranceRelative": f32(self.weld_tolerance_relative),
        }


def f32(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CookError("mesh numeric values must be real numbers, not booleans")
    if not math.isfinite(value):
        raise CookError("mesh contains a non-finite number")
    result = struct.unpack("<f", struct.pack("<f", value))[0]
    return 0.0 if result == 0 else result


def point_f32(point: Sequence[float]) -> Point:
    try:
        count = len(point)
    except TypeError as error:
        raise CookError("a point must be an array of three numbers") from error
    if count != 3:
        raise CookError("a point must have exactly three components")
    return (f32(point[0]), f32(point[1]), f32(point[2]))


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_source_uri(value: Any, label: str = "source URI") -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise CookError(f"{label} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CookError(f"{label} is not valid UTF-8") from error
    if len(encoded) > MAX_SOURCE_URI_BYTES:
        raise CookError(
            f"{label} exceeds {MAX_SOURCE_URI_BYTES} UTF-8 bytes"
        )
    return value


def parse_obj(data: bytes, source: str = "mesh.obj") -> Mesh:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CookError(f"{source}: OBJ must be UTF-8") from error

    vertices: list[Point] = []
    triangles: list[Triangle] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) < 4:
                raise CookError(f"{source}:{line_number}: malformed vertex")
            try:
                vertices.append(point_f32(tuple(float(x) for x in fields[1:4])))
            except ValueError as error:
                raise CookError(
                    f"{source}:{line_number}: malformed vertex number"
                ) from error
        elif fields[0] == "f":
            if len(fields) < 4:
                raise CookError(f"{source}:{line_number}: face has fewer than 3 vertices")
            face: list[int] = []
            for field in fields[1:]:
                token = field.split("/", 1)[0]
                try:
                    authored = int(token)
                except ValueError as error:
                    raise CookError(
                        f"{source}:{line_number}: malformed face index"
                    ) from error
                if authored == 0:
                    raise CookError(f"{source}:{line_number}: OBJ index zero is invalid")
                index = authored - 1 if authored > 0 else len(vertices) + authored
                if not 0 <= index < len(vertices):
                    raise CookError(f"{source}:{line_number}: face index out of range")
                face.append(index)
            # The cooker deliberately supports only triangles and validated
            # planar convex quads. Authored concave/non-planar caps must be
            # triangulated explicitly, avoiding a hidden geometry heuristic
            # in collision provenance.
            if len(face) > 4:
                raise CookError(
                    f"{source}:{line_number}: faces with more than four vertices "
                    "must be triangulated"
                )
            if len(set(face)) != len(face):
                raise CookError(f"{source}:{line_number}: face repeats a vertex")
            if len(face) == 4:
                points = [vertices[index] for index in face]
                relative = [subtract(point, points[0]) for point in points]
                normal = cross(relative[1], relative[2])
                normal_length = math.sqrt(dot(normal, normal))
                scale = max(
                    math.sqrt(dot(subtract(a, b), subtract(a, b)))
                    for a in points for b in points
                )
                if normal_length <= max(1.0e-20, scale * scale * 1.0e-12):
                    raise CookError(f"{source}:{line_number}: degenerate quad")
                if abs(dot(normal, relative[3])) > normal_length * scale * 1.0e-7:
                    raise CookError(f"{source}:{line_number}: non-planar quad")
                turn_tolerance = max(1.0e-30, scale ** 4 * 1.0e-12)
                for corner in range(4):
                    a = points[corner]
                    b = points[(corner + 1) % 4]
                    c = points[(corner + 2) % 4]
                    turn = dot(cross(subtract(b, a), subtract(c, b)), normal)
                    if turn <= turn_tolerance:
                        raise CookError(
                            f"{source}:{line_number}: quad is not strictly convex"
                        )
            for corner in range(1, len(face) - 1):
                triangle = (face[0], face[corner], face[corner + 1])
                if len(set(triangle)) != 3:
                    raise CookError(f"{source}:{line_number}: degenerate face indices")
                triangles.append(triangle)

    if len(vertices) < 4 or len(triangles) < 4:
        raise CookError(f"{source}: mesh must contain at least four vertices and faces")
    used = {index for triangle in triangles for index in triangle}
    if len(used) < 4:
        raise CookError(f"{source}: mesh references fewer than four unique vertices")
    return Mesh(tuple(vertices), tuple(triangles))


def scaled_mesh(mesh: Mesh, scale: Point) -> Mesh:
    scale = point_f32(scale)
    if any(component == 0 for component in scale):
        raise CookError("baked scale components must be nonzero")
    vertices = tuple(
        point_f32((point[0] * scale[0], point[1] * scale[1], point[2] * scale[2]))
        for point in mesh.vertices
    )
    triangles = mesh.triangles
    # A reflection reverses authored winding.  Component discovery is
    # orientation-insensitive, but preserving it gives CoACD a valid surface.
    if scale[0] * scale[1] * scale[2] < 0:
        triangles = tuple((a, c, b) for a, b, c in triangles)
    return Mesh(vertices, triangles)


def baked_mesh(mesh: Mesh, scale: Point, up_axis: str) -> Mesh:
    """Bake source-axis scale and a right-handed source-up to runtime-Z rotation."""
    if up_axis not in {"x", "y", "z"}:
        raise CookError("up axis must be x, y, or z")
    scaled = scaled_mesh(mesh, scale)

    def orient(point: Point) -> Point:
        x, y, z = point
        if up_axis == "z":
            return point
        if up_axis == "y":
            return point_f32((x, -z, y))
        # Match SurfaceMesh.load(upAxis: .x): a proper cyclic rotation that
        # maps the source +X up direction to runtime +Z without reflection.
        return point_f32((y, z, x))

    return Mesh(tuple(orient(point) for point in scaled.vertices), scaled.triangles)


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        # The lower representative wins so component identities do not depend
        # on dictionary/set iteration order.
        if a > b:
            a, b = b, a
        self.parent[b] = a


def mesh_diagonal(vertices: Sequence[Point]) -> float:
    lo = tuple(min(point[axis] for point in vertices) for axis in range(3))
    hi = tuple(max(point[axis] for point in vertices) for axis in range(3))
    return math.sqrt(sum((hi[axis] - lo[axis]) ** 2 for axis in range(3)))


def volume_validation_tolerance(diagonal: float, computed_volume: float) -> float:
    """Mirror the runtime's volume-dimensional Float32 error bound."""
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise CookError("convex hull scale is invalid")
    return f32(max(
        FLOAT32_MIN_NORMAL,
        diagonal * diagonal * diagonal * FLOAT32_EPSILON * 8.0,
        abs(computed_volume) * 5.0e-5,
    ))


def weld_vertices(
    vertices: Sequence[Point], relative: float, absolute: float
) -> tuple[list[int], float]:
    diagonal = mesh_diagonal(vertices)
    tolerance = max(diagonal * relative, absolute)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise CookError("computed weld tolerance is invalid")

    union = UnionFind(len(vertices))
    buckets: dict[tuple[int, int, int], list[int]] = {}
    ordered = sorted(range(len(vertices)), key=lambda index: (vertices[index], index))
    tolerance_squared = tolerance * tolerance
    for index in ordered:
        point = vertices[index]
        cell = tuple(math.floor(component / tolerance) for component in point)
        for offset in itertools.product((-1, 0, 1), repeat=3):
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            for candidate in buckets.get(neighbor, ()):
                other = vertices[candidate]
                distance_squared = sum(
                    (point[axis] - other[axis]) ** 2 for axis in range(3)
                )
                if distance_squared <= tolerance_squared:
                    union.union(index, candidate)
        buckets.setdefault(cell, []).append(index)
    return [union.find(index) for index in range(len(vertices))], tolerance


def connected_components(mesh: Mesh, parameters: CookParameters) -> list[Mesh]:
    welded, _ = weld_vertices(
        mesh.vertices,
        parameters.weld_tolerance_relative,
        parameters.weld_tolerance_absolute,
    )
    topology = UnionFind(len(mesh.vertices))
    for triangle in mesh.triangles:
        a, b, c = (welded[index] for index in triangle)
        topology.union(a, b)
        topology.union(a, c)

    grouped_triangles: dict[int, list[Triangle]] = {}
    for triangle in mesh.triangles:
        root = topology.find(welded[triangle[0]])
        if any(topology.find(welded[index]) != root for index in triangle):
            raise CookError("internal component discovery failure")
        grouped_triangles.setdefault(root, []).append(triangle)

    components: list[Mesh] = []
    for source_triangles in grouped_triangles.values():
        source_indices = sorted({index for tri in source_triangles for index in tri})
        remap = {source_index: index for index, source_index in enumerate(source_indices)}
        vertices = tuple(mesh.vertices[index] for index in source_indices)
        triangles = tuple(
            tuple(remap[index] for index in triangle)  # type: ignore[misc]
            for triangle in source_triangles
        )
        components.append(Mesh(vertices, triangles))

    def component_key(component: Mesh) -> tuple[Point, int, int]:
        return (min(component.vertices), len(component.vertices), len(component.triangles))

    components.sort(key=component_key)
    return components


def subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def multiply(a: Point, scalar: float) -> Point:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length_squared(a: Point) -> float:
    return dot(a, a)


def triangle_normal(vertices: Sequence[Point], triangle: Triangle) -> Point:
    a, b, c = (vertices[index] for index in triangle)
    return cross(subtract(b, a), subtract(c, a))


def oriented_face(
    a: int, b: int, c: int, vertices: Sequence[Point], interior: Point
) -> Triangle:
    triangle = (a, b, c)
    normal = triangle_normal(vertices, triangle)
    if dot(normal, subtract(interior, vertices[a])) > 0:
        return (a, c, b)
    return triangle


def incremental_convex_hull(points: Sequence[Point]) -> Mesh:
    vertices = sorted(set(point_f32(point) for point in points))
    if len(vertices) < 4:
        raise CookError("convex hull requires four unique vertices")
    diagonal = mesh_diagonal(vertices)
    tolerance = max(diagonal * 1.0e-7, 1.0e-9)

    a = 0
    b = max(
        range(1, len(vertices)),
        key=lambda index: (length_squared(subtract(vertices[index], vertices[a])), vertices[index]),
    )
    ab = subtract(vertices[b], vertices[a])
    ab_length_squared = length_squared(ab)
    c = max(
        (index for index in range(len(vertices)) if index not in {a, b}),
        key=lambda index: (
            length_squared(cross(subtract(vertices[index], vertices[a]), ab))
            / ab_length_squared,
            vertices[index],
        ),
    )
    normal = cross(subtract(vertices[b], vertices[a]), subtract(vertices[c], vertices[a]))
    normal_length = math.sqrt(length_squared(normal))
    if normal_length <= tolerance * tolerance:
        raise CookError("convex hull input is collinear")
    d = max(
        (index for index in range(len(vertices)) if index not in {a, b, c}),
        key=lambda index: (
            abs(dot(normal, subtract(vertices[index], vertices[a]))) / normal_length,
            vertices[index],
        ),
    )
    if abs(dot(normal, subtract(vertices[d], vertices[a]))) / normal_length <= tolerance:
        raise CookError("convex hull input is coplanar")

    interior = multiply(
        add(add(vertices[a], vertices[b]), add(vertices[c], vertices[d])), 0.25
    )
    faces = [
        oriented_face(a, b, c, vertices, interior),
        oriented_face(a, d, b, vertices, interior),
        oriented_face(a, c, d, vertices, interior),
        oriented_face(b, d, c, vertices, interior),
    ]
    seed = {a, b, c, d}
    for point_index in range(len(vertices)):
        if point_index in seed:
            continue
        point = vertices[point_index]
        visible: list[int] = []
        for face_index, face in enumerate(faces):
            face_normal = triangle_normal(vertices, face)
            normal_length = math.sqrt(length_squared(face_normal))
            distance = dot(face_normal, subtract(point, vertices[face[0]]))
            if distance > tolerance * normal_length:
                visible.append(face_index)
        if not visible:
            continue

        boundary: dict[tuple[int, int], tuple[int, int]] = {}
        counts: dict[tuple[int, int], int] = {}
        for face_index in visible:
            face = faces[face_index]
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted(edge))
                counts[key] = counts.get(key, 0) + 1
                boundary[key] = edge
        horizon = [boundary[key] for key, count in counts.items() if count == 1]
        if len(horizon) < 3:
            raise CookError("convex hull horizon was not a closed loop")
        visible_set = set(visible)
        faces = [face for index, face in enumerate(faces) if index not in visible_set]
        for u, v in sorted(horizon):
            faces.append(oriented_face(v, u, point_index, vertices, interior))

    used = sorted({index for face in faces for index in face})
    remap = {old: new for new, old in enumerate(used)}
    return Mesh(
        tuple(vertices[index] for index in used),
        tuple(tuple(remap[index] for index in face) for face in faces),  # type: ignore[arg-type]
    )


def repair_winding(vertices: Sequence[Point], triangles: Sequence[Triangle]) -> list[Triangle]:
    edge_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, triangle in enumerate(triangles):
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = (min(a, b), max(a, b))
            direction = 1 if (a, b) == key else -1
            edge_uses.setdefault(key, []).append((face_index, direction, 0))
    for edge, uses in edge_uses.items():
        if len(uses) != 2:
            raise CookError(f"generated hull is open/non-manifold at edge {edge}")

    orientation: list[Optional[int]] = [None] * len(triangles)
    orientation[0] = 1
    pending = [0]
    face_edges: list[list[tuple[tuple[int, int], int]]] = []
    for triangle in triangles:
        uses = []
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = (min(a, b), max(a, b))
            uses.append((key, 1 if (a, b) == key else -1))
        face_edges.append(uses)

    while pending:
        face = pending.pop()
        assert orientation[face] is not None
        for edge, direction in face_edges[face]:
            neighboring = edge_uses[edge]
            other, other_direction, _ = (
                neighboring[1] if neighboring[0][0] == face else neighboring[0]
            )
            required = -orientation[face] * direction * other_direction
            if orientation[other] is None:
                orientation[other] = required
                pending.append(other)
            elif orientation[other] != required:
                raise CookError("generated hull has inconsistent/non-orientable topology")
    if any(value is None for value in orientation):
        raise CookError("generated hull surface is disconnected")

    repaired = [
        triangle if orientation[index] == 1 else (triangle[0], triangle[2], triangle[1])
        for index, triangle in enumerate(triangles)
    ]
    lo = tuple(min(point[axis] for point in vertices) for axis in range(3))
    hi = tuple(max(point[axis] for point in vertices) for axis in range(3))
    reference = multiply(add(lo, hi), 0.5)
    volume6 = sum(
        dot(
            subtract(vertices[a], reference),
            cross(
                subtract(vertices[b], reference),
                subtract(vertices[c], reference),
            ),
        )
        for a, b, c in repaired
    )
    if not math.isfinite(volume6) or abs(volume6) <= 1.0e-18:
        raise CookError("generated hull has zero or non-finite volume")
    if volume6 < 0:
        repaired = [(a, c, b) for a, b, c in repaired]
    return repaired


def canonical_cycle(triangle: Triangle) -> Triangle:
    rotations = (
        triangle,
        (triangle[1], triangle[2], triangle[0]),
        (triangle[2], triangle[0], triangle[1]),
    )
    return min(rotations)


def validate_merged_face_loops(
    vertices: Sequence[Point], triangles: Sequence[Triangle]
) -> None:
    """Mirror the GPU uploader's deterministic coplanar-triangle merge.

    Collision clipping operates on maximal polygon faces rather than authored
    triangulation. Validate its fixed workspace limit offline and again at
    runtime so an otherwise valid 64-vertex hull cannot fail only on the GPU.
    """
    scale = max(mesh_diagonal(vertices), 1.0e-4)
    plane_tolerance = max(scale * 2.0e-6, 1.0e-7)
    normal_tolerance = 5.0e-5
    groups: list[dict[str, Any]] = []
    for triangle_index, triangle in enumerate(triangles):
        normal = triangle_normal(vertices, triangle)
        normal_length = math.sqrt(length_squared(normal))
        if not math.isfinite(normal_length) or normal_length <= 1.0e-12:
            raise CookError(f"generated hull triangle {triangle_index} is degenerate")
        normal = multiply(normal, 1.0 / normal_length)
        distance = dot(normal, vertices[triangle[0]])
        matching = None
        for index, group in enumerate(groups):
            if (
                dot(group["normal"], normal) > 0
                and math.sqrt(length_squared(cross(group["normal"], normal)))
                    <= normal_tolerance
                and abs(group["distance"] - distance) <= plane_tolerance
            ):
                matching = index
                break
        if matching is None:
            groups.append({
                "normal": normal,
                "distance": distance,
                "triangles": [triangle],
            })
        else:
            groups[matching]["triangles"].append(triangle)

    polygon_edges: dict[tuple[int, int], list[int]] = {}
    for group_index, group in enumerate(groups):
        counts: dict[tuple[int, int], int] = {}
        for triangle in group["triangles"]:
            for u, v in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = (min(u, v), max(u, v))
                counts[edge] = counts.get(edge, 0) + 1
        boundary = sorted(edge for edge, count in counts.items() if count == 1)
        if len(boundary) < 3:
            raise CookError(f"coplanar face {group_index} has no closed boundary")

        adjacency: dict[int, list[int]] = {}
        for a, b in boundary:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        for neighbors in adjacency.values():
            neighbors.sort()
        if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise CookError(f"coplanar face {group_index} is not one convex loop")
        start = min(adjacency)
        first = adjacency[start][0]
        loop = [start]
        previous, current = start, first
        while current != start and len(loop) <= len(boundary):
            loop.append(current)
            neighbors = adjacency.get(current)
            if not neighbors:
                break
            candidates = [neighbor for neighbor in neighbors if neighbor != previous]
            if not candidates:
                break
            previous, current = current, candidates[0]
        if (
            current != start
            or len(loop) != len(boundary)
            or len(set(loop)) != len(boundary)
        ):
            raise CookError(f"coplanar face {group_index} boundary is disconnected")
        if len(loop) > MAX_FACE_VERTICES:
            raise CookError(
                f"coplanar face {group_index} has {len(loop)} vertices; "
                f"runtime supports at most {MAX_FACE_VERTICES}"
            )
        for index, vertex in enumerate(loop):
            next_vertex = loop[(index + 1) % len(loop)]
            polygon_edges.setdefault(
                (min(vertex, next_vertex), max(vertex, next_vertex)), []
            ).append(group_index)

    if any(len(faces) != 2 for faces in polygon_edges.values()):
        raise CookError("merged polygon topology is not manifold")


def hull_digest(vertices: Sequence[Point], triangles: Sequence[Triangle]) -> str:
    payload = bytearray(b"avbd.convex-hull.v1\0")
    payload.extend(struct.pack("<I", len(vertices)))
    for vertex in vertices:
        payload.extend(struct.pack("<3f", *vertex))
    payload.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        payload.extend(struct.pack("<3I", *triangle))
    return sha256(bytes(payload))


def compound_digest(parts: Sequence[dict[str, Any]]) -> str:
    payload = bytearray(b"avbd.convex-compound.v1\0")
    payload.extend(struct.pack("<I", len(parts)))
    for part in parts:
        payload.extend(bytes.fromhex(part["digest"]))
    return sha256(bytes(payload))


def canonicalize_hull(
    raw_vertices: Sequence[Sequence[float]],
    raw_triangles: Sequence[Sequence[int]],
    max_vertices: int,
) -> dict[str, Any]:
    try:
        vertex_count = len(raw_vertices)
        triangle_count = len(raw_triangles)
    except TypeError as error:
        raise CookError("generated hull vertices and triangles must be arrays") from error
    allowed_vertices = min(max_vertices, MAX_VERTICES_PER_HULL)
    if not 4 <= vertex_count <= allowed_vertices:
        raise CookError(
            f"generated hull vertex count {vertex_count} is outside 4...{allowed_vertices}"
        )
    if not 4 <= triangle_count <= MAX_TRIANGLES_PER_HULL:
        raise CookError(
            "generated hull triangle count "
            f"{triangle_count} is outside 4...{MAX_TRIANGLES_PER_HULL}"
        )
    source_vertices = [point_f32(vertex) for vertex in raw_vertices]

    unique_vertices = sorted(set(source_vertices))
    old_to_point = {index: source_vertices[index] for index in range(len(source_vertices))}
    point_to_index = {point: index for index, point in enumerate(unique_vertices)}
    triangles: list[Triangle] = []
    for raw_triangle in raw_triangles:
        try:
            index_count = len(raw_triangle)
        except TypeError as error:
            raise CookError("generated hull face must be an index array") from error
        if index_count != 3:
            raise CookError("generated hull face is not a triangle")
        if any(
            isinstance(index, bool) or not isinstance(index, numbers.Integral)
            for index in raw_triangle
        ):
            raise CookError("generated hull triangle indices must be integers")
        triangle = tuple(int(index) for index in raw_triangle)
        if any(index < 0 or index >= len(source_vertices) for index in triangle):
            raise CookError("generated hull triangle index is out of range")
        remapped = tuple(point_to_index[old_to_point[index]] for index in triangle)
        if len(set(remapped)) != 3:
            raise CookError("generated hull contains a collapsed triangle")
        triangles.append(remapped)  # type: ignore[arg-type]
    if len(triangles) < 4:
        raise CookError("generated hull has fewer than four triangles")
    if len({tuple(sorted(triangle)) for triangle in triangles}) != len(triangles):
        raise CookError("generated hull contains duplicate triangles")

    used = sorted({index for triangle in triangles for index in triangle})
    vertices = [unique_vertices[index] for index in used]
    remap = {old: new for new, old in enumerate(used)}
    triangles = [
        tuple(remap[index] for index in triangle)  # type: ignore[misc]
        for triangle in triangles
    ]
    if len(vertices) > max_vertices:
        raise CookError(
            f"generated hull has {len(vertices)} vertices, exceeding "
            f"max vertices per hull {max_vertices}"
        )
    triangles = repair_winding(vertices, triangles)
    triangles = sorted(canonical_cycle(triangle) for triangle in triangles)
    validate_merged_face_loops(vertices, triangles)

    diagonal = mesh_diagonal(vertices)
    tolerance = max(diagonal * 1.0e-5, 1.0e-7)
    for face_index, triangle in enumerate(triangles):
        normal = triangle_normal(vertices, triangle)
        normal_length = math.sqrt(length_squared(normal))
        if normal_length <= tolerance * tolerance:
            raise CookError(f"generated hull face {face_index} is degenerate")
        anchor = vertices[triangle[0]]
        for point in vertices:
            signed_distance = dot(normal, subtract(point, anchor)) / normal_length
            if signed_distance > tolerance:
                raise CookError(
                    f"generated part is not convex at face {face_index} "
                    f"(distance {signed_distance})"
                )

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(triangles):
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(face_index)
    edges = []
    for (a, b), faces in sorted(edge_faces.items()):
        if len(faces) != 2:
            raise CookError(f"generated hull is open/non-manifold at edge {(a, b)}")
        face_a, face_b = sorted(faces)
        edges.append(
            {"faceA": face_a, "faceB": face_b, "vertexA": a, "vertexB": b}
        )
    if len(edges) > MAX_EDGES_PER_HULL:
        raise CookError(
            f"generated hull has {len(edges)} edges, exceeding {MAX_EDGES_PER_HULL}"
        )

    lo = tuple(min(point[axis] for point in vertices) for axis in range(3))
    hi = tuple(max(point[axis] for point in vertices) for axis in range(3))
    bounds_center = multiply(add(lo, hi), 0.5)
    radius = max(math.sqrt(length_squared(subtract(point, bounds_center))) for point in vertices)

    volume6 = 0.0
    local_centroid_numerator = (0.0, 0.0, 0.0)
    volume_reference = bounds_center
    for a, b, c in triangles:
        local_a = subtract(vertices[a], volume_reference)
        local_b = subtract(vertices[b], volume_reference)
        local_c = subtract(vertices[c], volume_reference)
        tetra6 = dot(local_a, cross(local_b, local_c))
        volume6 += tetra6
        local_centroid_numerator = add(
            local_centroid_numerator,
            multiply(add(add(local_a, local_b), local_c), tetra6),
        )
    if volume6 <= 0 or not math.isfinite(volume6):
        raise CookError("generated hull does not have positive finite volume")
    centroid = add(
        volume_reference,
        multiply(local_centroid_numerator, 1.0 / (4.0 * volume6)),
    )

    vertices = [point_f32(point) for point in vertices]
    digest = hull_digest(vertices, triangles)
    return {
        "boundingRadius": f32(radius),
        "boundsMax": list(point_f32(hi)),
        "boundsMin": list(point_f32(lo)),
        "centroid": list(point_f32(centroid)),
        "digest": digest,
        "edges": edges,
        "stableID": f"hull-{digest[:16]}",
        "triangles": [list(triangle) for triangle in triangles],
        "vertices": [list(point) for point in vertices],
        "volume": f32(volume6 / 6.0),
    }


def geometry_digest(mesh: Mesh) -> str:
    canonical_triangles = []
    for triangle in mesh.triangles:
        points = sorted(mesh.vertices[index] for index in triangle)
        canonical_triangles.append(tuple(points))
    canonical_triangles.sort()
    payload = bytearray(b"avbd.source-triangles.v1\0")
    payload.extend(struct.pack("<I", len(canonical_triangles)))
    for triangle in canonical_triangles:
        for point in triangle:
            payload.extend(struct.pack("<3f", *point))
    return sha256(bytes(payload))


def load_coacd_backend() -> tuple[Any, str]:
    try:
        module = importlib.import_module("coacd")
    except (ImportError, ModuleNotFoundError) as error:
        raise CookError(
            "CoACD is unavailable. Install the pinned offline cooker dependency "
            "or explicitly request --method hull; no implicit fallback is allowed."
        ) from error
    try:
        version = importlib.metadata.version("coacd")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version.strip():
        raise CookError("CoACD backend does not expose an exact version")
    version = version.strip()
    if version != PINNED_COACD_VERSION:
        raise CookError(
            f"CoACD {PINNED_COACD_VERSION} is required for reproducible cooking; "
            f"found {version}"
        )
    return module, version


def coacd_decompose(
    component: Mesh,
    parameters: CookParameters,
    module: Any,
) -> list[tuple[Sequence[Sequence[float]], Sequence[Sequence[int]]]]:
    try:
        import numpy as np
    except ImportError as error:
        raise CookError("CoACD cooking requires NumPy") from error
    mesh = module.Mesh(
        np.asarray(component.vertices, dtype=np.float64),
        np.asarray(component.triangles, dtype=np.int32),
    )
    # CoACD's ``max_convex_hull`` is the number of output parts (and only
    # affects merge); ``max_ch_vertex`` is the per-part vertex cap.  Keeping
    # these distinct is essential -- Newton's historical helper mapped the
    # runtime vertex cap to the former option.
    kwargs = {
        "decimate": parameters.coacd_decimate,
        "extrude": parameters.coacd_extrude,
        "extrude_margin": parameters.coacd_extrude_margin,
        "max_ch_vertex": parameters.max_vertices_per_hull,
        "max_convex_hull": parameters.max_hulls,
        "mcts_iterations": parameters.coacd_mcts_iterations,
        "mcts_max_depth": parameters.coacd_mcts_max_depth,
        "mcts_nodes": parameters.coacd_mcts_nodes,
        "merge": parameters.coacd_merge,
        "pca": parameters.coacd_pca,
        "apx_mode": parameters.coacd_approximation_mode,
        "preprocess_mode": parameters.coacd_preprocess_mode,
        "preprocess_resolution": parameters.coacd_preprocess_resolution,
        "real_metric": True,
        "resolution": parameters.coacd_resolution,
        "seed": parameters.seed,
        "threshold": parameters.threshold_meters,
    }
    try:
        output = module.run_coacd(mesh, **kwargs)
    except Exception as error:
        raise CookError(f"CoACD failed: {error}") from error
    if not output:
        raise CookError("CoACD returned no convex parts")
    return list(output)


def cache_key(
    source_sha256: str,
    source_byte_count: int,
    source_uri: str,
    geometry_sha256: str,
    scale: Point,
    up_axis: str,
    method: str,
    backend: str,
    backend_version: str,
    parameters: CookParameters,
) -> str:
    """Cross-language, source-specific cache/provenance seal.

    Swift implements the same fixed little-endian contract. Keeping this
    binary avoids JSON encoder differences for Float32 spellings.
    """
    parameters.validate()
    if not is_lowercase_sha256(source_sha256):
        raise CookError("source cache-key SHA-256 is invalid")
    if not is_lowercase_sha256(geometry_sha256):
        raise CookError("geometry cache-key SHA-256 is invalid")
    if type(source_byte_count) is not int or not 0 < source_byte_count <= 0x7FFF_FFFF_FFFF_FFFF:
        raise CookError("source cache-key byte count is invalid")
    validate_source_uri(source_uri, "source cache-key URI")
    if up_axis not in {"x", "y", "z"}:
        raise CookError("source cache-key up axis is invalid")
    if method == "hull":
        if backend != BUILTIN_HULL_BACKEND or backend_version != BUILTIN_HULL_VERSION:
            raise CookError("builtin hull cache-key identity is not pinned")
    elif method == "coacd":
        if backend != "coacd-python" or backend_version != PINNED_COACD_VERSION:
            raise CookError("CoACD cache-key identity is not pinned")
    else:
        raise CookError("cache-key algorithm is invalid")

    payload = bytearray(b"avbd.convex-cook-key.v1\0")

    def append_string(value: str) -> None:
        encoded = value.encode("utf-8")
        if len(encoded) > 0xFFFF_FFFF:
            raise CookError("cache-key string is too long")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)

    def append_bool(value: bool) -> None:
        if type(value) is not bool:
            raise CookError("cache-key boolean has the wrong type")
        payload.extend(b"\x01" if value else b"\x00")

    payload.extend(bytes.fromhex(source_sha256))
    payload.extend(struct.pack("<Q", source_byte_count))
    append_string(source_uri)
    payload.extend(bytes.fromhex(geometry_sha256))
    payload.extend(struct.pack("<3f", *point_f32(scale)))
    append_string(up_axis)
    append_string(method)
    append_string(backend)
    append_string(backend_version)
    payload.extend(struct.pack("<I", COOKER_CONTRACT_VERSION))
    payload.extend(struct.pack("<f", f32(parameters.threshold_meters)))
    payload.extend(struct.pack("<I", parameters.max_hulls))
    payload.extend(struct.pack("<I", parameters.max_vertices_per_hull))
    payload.extend(struct.pack("<I", parameters.seed))
    payload.extend(struct.pack("<f", f32(parameters.weld_tolerance_relative)))
    payload.extend(struct.pack("<f", f32(parameters.weld_tolerance_absolute)))
    append_bool(parameters.split_connected_components)
    payload.extend(struct.pack("<I", parameters.coacd_mcts_nodes))
    payload.extend(struct.pack("<I", parameters.coacd_mcts_iterations))
    payload.extend(struct.pack("<I", parameters.coacd_mcts_max_depth))
    append_string(parameters.coacd_preprocess_mode)
    payload.extend(struct.pack("<I", parameters.coacd_preprocess_resolution))
    payload.extend(struct.pack("<I", parameters.coacd_resolution))
    append_bool(parameters.coacd_merge)
    append_bool(parameters.coacd_decimate)
    append_bool(parameters.coacd_extrude)
    payload.extend(struct.pack("<f", f32(parameters.coacd_extrude_margin)))
    append_bool(parameters.coacd_pca)
    append_string(parameters.coacd_approximation_mode)
    append_bool(True)  # CoACD real-metric contract
    return sha256(bytes(payload))


def cook_asset(
    source_data: bytes,
    source_uri: str,
    *,
    method: str = "coacd",
    scale: Point = (1.0, 1.0, 1.0),
    up_axis: str = "z",
    parameters: CookParameters = CookParameters(),
    coacd_backend: Optional[Any] = None,
    coacd_backend_version: Optional[str] = None,
) -> dict[str, Any]:
    parameters.validate()
    if method not in {"coacd", "hull"}:
        raise CookError("method must be coacd or hull")
    if up_axis not in {"x", "y", "z"}:
        raise CookError("up axis must be x, y, or z")
    validate_source_uri(source_uri)

    source_mesh = parse_obj(source_data, source_uri)
    baked_scale = point_f32(scale)
    mesh = baked_mesh(source_mesh, baked_scale, up_axis)
    components = connected_components(mesh, parameters)
    if len(components) > parameters.max_hulls:
        raise CookError(
            f"source has {len(components)} connected components, exceeding "
            f"max hulls {parameters.max_hulls}"
        )

    if method == "coacd":
        if coacd_backend is None:
            coacd_backend, coacd_backend_version = load_coacd_backend()
        if coacd_backend_version != PINNED_COACD_VERSION:
            raise CookError(
                f"CoACD {PINNED_COACD_VERSION} is required for reproducible cooking; "
                f"found {coacd_backend_version!r}"
            )
        backend = "coacd-python"
        backend_version = coacd_backend_version
    else:
        if coacd_backend is not None or coacd_backend_version is not None:
            raise CookError("CoACD backend overrides are invalid for hull cooking")
        backend = BUILTIN_HULL_BACKEND
        backend_version = BUILTIN_HULL_VERSION

    parts = []
    for component in components:
        if method == "hull":
            hull = incremental_convex_hull(component.vertices)
            generated = [(hull.vertices, hull.triangles)]
        else:
            generated = coacd_decompose(component, parameters, coacd_backend)
        for vertices, triangles in generated:
            parts.append(
                canonicalize_hull(
                    vertices, triangles, parameters.max_vertices_per_hull
                )
            )
    parts.sort(key=lambda part: (part["digest"], part["stableID"]))
    if not parts or len(parts) > parameters.max_hulls:
        raise CookError(
            f"cooker produced {len(parts)} parts; allowed range is "
            f"1...{parameters.max_hulls}"
        )
    if len({part["digest"] for part in parts}) != len(parts):
        raise CookError("cooker produced duplicate convex parts")

    geometry_sha256 = geometry_digest(mesh)
    source_sha256 = sha256(source_data)
    key = cache_key(
        source_sha256,
        len(source_data),
        source_uri,
        geometry_sha256,
        baked_scale,
        up_axis,
        method,
        backend,
        backend_version,
        parameters,
    )
    asset = {
        "cacheKey": key,
        "cooker": {
            "algorithm": method,
            "backend": backend,
            "backendVersion": backend_version,
            "parameters": parameters.as_json(),
        },
        "digest": compound_digest(parts),
        "kind": ASSET_KIND,
        "parts": parts,
        "schemaVersion": SCHEMA_VERSION,
        "source": {
            "bakedScale": list(baked_scale),
            "byteCount": len(source_data),
            "geometrySHA256": geometry_sha256,
            "sha256": source_sha256,
            "upAxis": up_axis,
            "uri": source_uri,
        },
    }
    validate_asset(asset)
    return asset


def validate_asset(asset: dict[str, Any]) -> None:
    """Validate untrusted decoded JSON and expose only stable CookError failures."""
    try:
        _validate_asset(asset)
    except CookError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError,
            struct.error) as error:
        raise CookError(f"malformed convex asset: {error}") from error


def _validate_asset(asset: dict[str, Any]) -> None:
    expect_keys(
        asset,
        {"cacheKey", "cooker", "digest", "kind", "parts", "schemaVersion", "source"},
        "asset",
    )
    if type(asset.get("schemaVersion")) is not int:
        raise CookError("asset schema version must be an integer")
    if asset["schemaVersion"] != SCHEMA_VERSION:
        raise CookError("asset schema version is unsupported")
    if asset.get("kind") != ASSET_KIND:
        raise CookError("asset kind is unsupported")
    for field in ("cacheKey", "digest"):
        value = asset.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise CookError(f"asset {field} is not a lowercase SHA-256")
    source = asset.get("source")
    expect_keys(
        source,
        {"bakedScale", "byteCount", "geometrySHA256", "sha256", "upAxis", "uri"},
        "asset source",
    )
    validate_source_uri(source["uri"], "asset source URI")
    if (
        not isinstance(source["byteCount"], int)
        or isinstance(source["byteCount"], bool)
        or not 0 < source["byteCount"] <= 0x7FFF_FFFF_FFFF_FFFF
    ):
        raise CookError("asset source byte count must be positive")
    for field in ("sha256", "geometrySHA256"):
        if not is_lowercase_sha256(source[field]):
            raise CookError(f"asset source {field} is not a lowercase SHA-256")
    if source["upAxis"] not in {"x", "y", "z"}:
        raise CookError("asset source up axis is invalid")
    baked_scale = point_f32(source["bakedScale"])
    if any(component == 0 for component in baked_scale):
        raise CookError("asset source baked scale is invalid")

    cooker = asset.get("cooker")
    expect_keys(cooker, {"algorithm", "backend", "backendVersion", "parameters"},
                "asset cooker")
    if type(cooker["algorithm"]) is not str or cooker["algorithm"] not in {"coacd", "hull"}:
        raise CookError("asset cooker algorithm is invalid")
    if not isinstance(cooker["backend"], str) or not cooker["backend"]:
        raise CookError("asset cooker backend is invalid")
    if not isinstance(cooker["backendVersion"], str) or not cooker["backendVersion"]:
        raise CookError("asset cooker backend version is invalid")
    if cooker["algorithm"] == "hull" and (
        cooker["backend"] != BUILTIN_HULL_BACKEND
        or cooker["backendVersion"] != BUILTIN_HULL_VERSION
    ):
        raise CookError("builtin hull cooker identity is not pinned")
    if cooker["algorithm"] == "coacd" and (
        cooker["backend"] != "coacd-python"
        or cooker["backendVersion"] != PINNED_COACD_VERSION
    ):
        raise CookError("CoACD cooker identity is not pinned")
    parameters = parameters_from_json(cooker["parameters"])

    parts = asset.get("parts")
    if not isinstance(parts, list) or not parts:
        raise CookError("asset has no convex parts")
    if len(parts) > MAX_HULLS or len(parts) > parameters.max_hulls:
        raise CookError("asset part count exceeds its max hulls metadata")
    part_keys = {
        "boundingRadius", "boundsMax", "boundsMin", "centroid", "digest",
        "edges", "stableID", "triangles", "vertices", "volume",
    }
    for index, part in enumerate(parts):
        expect_keys(part, part_keys, f"convex part {index}")
        if not isinstance(part["vertices"], list):
            raise CookError(f"convex part {index} vertices must be an array")
        if not 4 <= len(part["vertices"]) <= MAX_VERTICES_PER_HULL:
            raise CookError(f"convex part {index} vertex count is out of bounds")
        if not isinstance(part["triangles"], list):
            raise CookError(f"convex part {index} triangles must be an array")
        if not 4 <= len(part["triangles"]) <= MAX_TRIANGLES_PER_HULL:
            raise CookError(f"convex part {index} triangle count is out of bounds")
        if not isinstance(part["edges"], list):
            raise CookError(f"convex part {index} edges must be an array")
        if not 6 <= len(part["edges"]) <= MAX_EDGES_PER_HULL:
            raise CookError(f"convex part {index} edge count is out of bounds")
        if not is_lowercase_sha256(part["digest"]):
            raise CookError(f"convex part {index} digest is invalid")
        if type(part["stableID"]) is not str:
            raise CookError(f"convex part {index} stableID must be a string")
    digests = [part["digest"] for part in parts]
    if len(set(digests)) != len(digests):
        raise CookError("convex compound contains duplicate parts")
    if asset["digest"] != compound_digest(parts):
        raise CookError("compound digest does not match its parts")
    if parts != sorted(parts, key=lambda part: (part["digest"], part["stableID"])):
        raise CookError("convex parts are not canonically sorted")
    for part in parts:
        canonical = canonicalize_hull(
            part["vertices"],
            part["triangles"],
            parameters.max_vertices_per_hull,
        )
        stored_volume = f32(part["volume"])
        expected_volume = canonical["volume"]
        volume_tolerance = volume_validation_tolerance(
            mesh_diagonal([tuple(vertex) for vertex in canonical["vertices"]]),
            expected_volume,
        )
        if abs(stored_volume - expected_volume) > volume_tolerance:
            raise CookError(
                f"convex part {part.get('stableID')} stored volume is incorrect"
            )
        if canonical_json_bytes(part) != canonical_json_bytes(canonical):
            raise CookError(f"convex part {part.get('stableID')} is not canonical")

    expected_cache_key = cache_key(
        source["sha256"],
        source["byteCount"],
        source["uri"],
        source["geometrySHA256"],
        baked_scale,
        source["upAxis"],
        cooker["algorithm"],
        cooker["backend"],
        cooker["backendVersion"],
        parameters,
    )
    if asset["cacheKey"] != expected_cache_key:
        raise CookError("asset cache key does not match source/cooker metadata")


def is_lowercase_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def expect_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise CookError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CookError(f"{label} keys differ (missing={missing}, extra={extra})")


def parameters_from_json(value: Any) -> CookParameters:
    expected = {
        "coacdApproximationMode",
        "coacdDecimate",
        "coacdExtrude",
        "coacdExtrudeMargin",
        "coacdMCTSIterations",
        "coacdMCTSMaxDepth",
        "coacdMCTSNodes",
        "coacdMerge",
        "coacdPCA",
        "coacdPreprocessMode",
        "coacdPreprocessResolution",
        "coacdRealMetric",
        "coacdResolution",
        "maxHulls",
        "maxVerticesPerHull",
        "seed",
        "splitConnectedComponents",
        "thresholdMeters",
        "weldToleranceAbsolute",
        "weldToleranceRelative",
    }
    expect_keys(value, expected, "asset cooker parameters")
    boolean_fields = (
        "coacdDecimate", "coacdExtrude", "coacdMerge", "coacdPCA",
        "coacdRealMetric", "splitConnectedComponents",
    )
    if any(not isinstance(value[field], bool) for field in boolean_fields):
        raise CookError("asset cooker boolean parameter has the wrong type")
    integer_fields = (
        "coacdMCTSIterations", "coacdMCTSMaxDepth", "coacdMCTSNodes",
        "coacdPreprocessResolution", "coacdResolution", "maxHulls",
        "maxVerticesPerHull", "seed",
    )
    if any(
        not isinstance(value[field], int) or isinstance(value[field], bool)
        for field in integer_fields
    ):
        raise CookError("asset cooker integer parameter has the wrong type")
    if not value["coacdRealMetric"]:
        raise CookError("asset cooker must use CoACD real-metric mode")
    parameters = CookParameters(
        threshold_meters=value["thresholdMeters"],
        max_hulls=value["maxHulls"],
        max_vertices_per_hull=value["maxVerticesPerHull"],
        seed=value["seed"],
        weld_tolerance_relative=value["weldToleranceRelative"],
        weld_tolerance_absolute=value["weldToleranceAbsolute"],
        split_connected_components=value["splitConnectedComponents"],
        coacd_mcts_nodes=value["coacdMCTSNodes"],
        coacd_mcts_iterations=value["coacdMCTSIterations"],
        coacd_mcts_max_depth=value["coacdMCTSMaxDepth"],
        coacd_preprocess_mode=value["coacdPreprocessMode"],
        coacd_preprocess_resolution=value["coacdPreprocessResolution"],
        coacd_resolution=value["coacdResolution"],
        coacd_merge=value["coacdMerge"],
        coacd_decimate=value["coacdDecimate"],
        coacd_extrude=value["coacdExtrude"],
        coacd_extrude_margin=value["coacdExtrudeMargin"],
        coacd_pca=value["coacdPCA"],
        coacd_approximation_mode=value["coacdApproximationMode"],
    )
    parameters.validate()
    if canonical_json_bytes(parameters.as_json()) != canonical_json_bytes(value):
        raise CookError("asset cooker parameters are not canonical Float32 values")
    return parameters


def load_canonical_asset(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        encoded_size = path.stat().st_size
    except OSError as error:
        raise CookError(f"{path}: could not inspect asset: {error}") from error
    if encoded_size > MAX_ASSET_BYTES:
        raise CookError(
            f"{path}: asset is {encoded_size} bytes; maximum is {MAX_ASSET_BYTES}"
        )
    data = path.read_bytes()
    if len(data) > MAX_ASSET_BYTES:
        raise CookError(
            f"{path}: asset is {len(data)} bytes; maximum is {MAX_ASSET_BYTES}"
        )

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CookError(f"asset contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        asset = json.loads(data, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, CookError) as error:
        raise CookError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(asset, dict):
        raise CookError(f"{path}: asset root must be an object")
    try:
        if canonical_json_bytes(asset) != data:
            raise CookError(f"{path}: JSON is not canonical")
        validate_asset(asset)
    except CookError:
        raise
    except (TypeError, ValueError, OverflowError, UnicodeError) as error:
        raise CookError(f"{path}: malformed convex asset: {error}") from error
    return asset, data


def verify_asset_file(
    path: Path,
    source_path: Optional[Path] = None,
    debug_obj_path: Optional[Path] = None,
) -> dict[str, Any]:
    asset, _ = load_canonical_asset(path)
    if source_path is not None:
        source_data = source_path.read_bytes()
        source = asset["source"]
        if len(source_data) != source["byteCount"]:
            raise CookError("source byte count does not match the asset")
        if sha256(source_data) != source["sha256"]:
            raise CookError("source SHA-256 does not match the asset")
        source_mesh = parse_obj(source_data, source["uri"])
        mesh = baked_mesh(
            source_mesh, tuple(source["bakedScale"]), source["upAxis"]
        )
        if geometry_digest(mesh) != source["geometrySHA256"]:
            raise CookError("source canonical geometry SHA-256 does not match the asset")
    if debug_obj_path is not None:
        actual_debug = debug_obj_path.read_bytes()
        expected_debug = debug_obj(asset)
        if actual_debug != expected_debug:
            raise CookError("debug OBJ does not match the canonical convex asset")
    return asset


def debug_obj(asset: dict[str, Any]) -> bytes:
    lines = ["# Deterministic AVBD convex-compound debug mesh"]
    vertex_base = 1
    for part in asset["parts"]:
        lines.append(f"o {part['stableID']}")
        for x, y, z in part["vertices"]:
            lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
        for a, b, c in part["triangles"]:
            lines.append(
                f"f {vertex_base + a} {vertex_base + b} {vertex_base + c}"
            )
        vertex_base += len(part["vertices"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_many(
    outputs: Sequence[tuple[Path, bytes]],
    protected_paths: Sequence[Path] = (),
) -> None:
    """Stage outputs atomically without overwriting any protected source."""
    normalized = [(path.resolve(), data) for path, data in outputs]
    paths = [path for path, _ in normalized]
    if len(set(paths)) != len(paths):
        raise CookError("output, debug OBJ, and cache paths must be distinct")
    protected = [path.resolve() for path in protected_paths]
    for destination in paths:
        for source in protected:
            aliases = destination == source
            if not aliases and destination.exists() and source.exists():
                aliases = os.path.samefile(destination, source)
            if aliases:
                raise CookError("an output path aliases the input source")

    staged: dict[Path, str] = {}
    backups: dict[Path, str] = {}
    published: list[Path] = []
    try:
        for path, data in normalized:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            try:
                os.fchmod(descriptor, 0o644)
                with os.fdopen(descriptor, "wb") as file:
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
            staged[path] = temporary_name

        for path, _ in normalized:
            if path.exists():
                backup_descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{path.name}.", suffix=".backup", dir=path.parent
                )
                os.close(backup_descriptor)
                os.unlink(backup_name)
                os.replace(path, backup_name)
                backups[path] = backup_name
            staged_name = staged[path]
            os.replace(staged_name, path)
            del staged[path]
            published.append(path)
    except BaseException:
        for path in reversed(published):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        for path, backup_name in backups.items():
            try:
                os.replace(backup_name, path)
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_name in staged.values():
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        for backup_name in backups.values():
            try:
                os.unlink(backup_name)
            except FileNotFoundError:
                pass


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path,
                        help="source OBJ (optional source check in --verify mode)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path,
                        help="validate a cooked asset without loading CoACD")
    parser.add_argument("--debug-obj", type=Path)
    parser.add_argument("--source-uri", help="stable source identifier; defaults to basename")
    parser.add_argument("--method", choices=("coacd", "hull"), default="coacd")
    parser.add_argument("--scale", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--threshold-meters", type=float, default=0.05)
    parser.add_argument("--max-parts", type=int, default=64)
    parser.add_argument(
        "--max-vertices-per-hull", type=int, default=DEFAULT_MAX_VERTICES_PER_HULL
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weld-relative", type=float, default=1.0e-7)
    parser.add_argument("--weld-absolute", type=float, default=1.0e-9)
    parser.add_argument("--coacd-mcts-nodes", type=int, default=20)
    parser.add_argument("--coacd-mcts-iterations", type=int, default=150)
    parser.add_argument("--coacd-mcts-max-depth", type=int, default=3)
    parser.add_argument(
        "--coacd-preprocess-mode", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--coacd-preprocess-resolution", type=int, default=50)
    parser.add_argument("--coacd-resolution", type=int, default=2_000)
    parser.add_argument(
        "--cache-directory",
        type=Path,
        help="optional content-addressed cache (ordinary builds never require it)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(tokens)
    try:
        if args.verify is not None:
            if args.output is not None:
                raise CookError("--verify cannot be combined with --output")
            cook_only_options = {
                "--source-uri", "--method", "--scale", "--up-axis",
                "--threshold-meters", "--max-parts", "--max-vertices-per-hull",
                "--seed", "--weld-relative", "--weld-absolute",
                "--coacd-mcts-nodes", "--coacd-mcts-iterations",
                "--coacd-mcts-max-depth", "--coacd-preprocess-mode",
                "--coacd-preprocess-resolution", "--coacd-resolution",
                "--cache-directory",
            }
            supplied = {token.split("=", 1)[0] for token in tokens}
            ignored = sorted(cook_only_options.intersection(supplied))
            if ignored:
                raise CookError(
                    "--verify cannot be combined with cook-only option "
                    + ignored[0]
                )
            verify_asset_file(args.verify, args.input, args.debug_obj)
            return 0
        if args.input is None or args.output is None:
            raise CookError("cooking requires both --input and --output")
        source_data = args.input.read_bytes()
        parameters = CookParameters(
            threshold_meters=args.threshold_meters,
            max_hulls=args.max_parts,
            max_vertices_per_hull=args.max_vertices_per_hull,
            seed=args.seed,
            weld_tolerance_relative=args.weld_relative,
            weld_tolerance_absolute=args.weld_absolute,
            coacd_mcts_nodes=args.coacd_mcts_nodes,
            coacd_mcts_iterations=args.coacd_mcts_iterations,
            coacd_mcts_max_depth=args.coacd_mcts_max_depth,
            coacd_preprocess_mode=args.coacd_preprocess_mode,
            coacd_preprocess_resolution=args.coacd_preprocess_resolution,
            coacd_resolution=args.coacd_resolution,
        )
        asset = cook_asset(
            source_data,
            args.source_uri or args.input.name,
            method=args.method,
            scale=tuple(args.scale),
            up_axis=args.up_axis,
            parameters=parameters,
        )
        encoded = canonical_json_bytes(asset)
        outputs: list[tuple[Path, bytes]] = []
        if args.cache_directory:
            cached = args.cache_directory / f"{asset['cacheKey']}.avbdconvex.json"
            if cached.exists():
                existing = cached.read_bytes()
                if existing != encoded:
                    raise CookError(f"cache collision or corrupt entry: {cached}")
            outputs.append((cached, encoded))
        outputs.append((args.output, encoded))
        if args.debug_obj:
            outputs.append((args.debug_obj, debug_obj(asset)))
        atomic_write_many(outputs, protected_paths=(args.input,))
    except (CookError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
