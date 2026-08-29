#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import cook_convex_asset as cooker
import normalize_classic_obj as normalizer


TETRA_OBJ = b"""\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""

TINY_TETRA_OBJ = b"""\
v 0 0 0
v 0.0001 0 0
v 0 0.0001 0
v 0 0 0.0001
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""

TRANSLATED_TETRA_OBJ = b"""\
v 1000000 1000000 1000000
v 1000001 1000000 1000000
v 1000000 1000001 1000000
v 1000000 1000000 1000001
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def two_tetra_obj(offset: float = 3.0) -> bytes:
    return (
        TETRA_OBJ.decode("utf-8")
        + f"""\
v {offset} 0 0
v {offset + 1} 0 0
v {offset} 1 0
v {offset} 0 1
f 5 7 6
f 5 6 8
f 5 8 7
f 6 7 8
"""
    ).encode("utf-8")


def regular_polygon_prism(sides: int):
    lower = [
        (cooker.math.cos(2 * cooker.math.pi * index / sides),
         cooker.math.sin(2 * cooker.math.pi * index / sides), -0.5)
        for index in range(sides)
    ]
    upper = [(x, y, 0.5) for x, y, _ in lower]
    triangles = []
    for index in range(1, sides - 1):
        triangles.append((0, index + 1, index))
        triangles.append((sides, sides + index, sides + index + 1))
    for index in range(sides):
        next_index = (index + 1) % sides
        triangles.append((index, next_index, sides + next_index))
        triangles.append((index, sides + next_index, sides + index))
    return lower + upper, triangles


class FakeCoACD:
    __version__ = cooker.PINNED_COACD_VERSION

    def __init__(self) -> None:
        self.calls = []

    @staticmethod
    def Mesh(vertices, triangles):
        return vertices, triangles

    def run_coacd(self, mesh, **kwargs):
        self.calls.append(kwargs)
        vertices, triangles = mesh
        return [(vertices, triangles)]


class ConvexAssetCookerTests(unittest.TestCase):
    def test_classic_normalization_is_byte_deterministic_and_grounded(self) -> None:
        parsed = cooker.parse_obj(TRANSLATED_TETRA_OBJ, "translated.obj")
        first, first_offset = normalizer.normalized_mesh(parsed, 0.5, "y")
        second, second_offset = normalizer.normalized_mesh(parsed, 0.5, "y")
        self.assertEqual(first, second)
        self.assertEqual(first_offset, second_offset)
        self.assertAlmostEqual(min(point[2] for point in first.vertices), 0.0)
        self.assertAlmostEqual(
            min(point[0] for point in first.vertices)
            + max(point[0] for point in first.vertices),
            0.0,
        )
        self.assertAlmostEqual(
            min(point[1] for point in first.vertices)
            + max(point[1] for point in first.vertices),
            0.0,
        )
        source_hash = hashlib.sha256(TRANSLATED_TETRA_OBJ).hexdigest()
        encoded = normalizer.encode_obj(
            first,
            label="Fixture",
            source_sha256=source_hash,
            scale=0.5,
            up_axis="y",
            offset=first_offset,
        )
        self.assertEqual(
            encoded,
            normalizer.encode_obj(
                second,
                label="Fixture",
                source_sha256=source_hash,
                scale=0.5,
                up_axis="y",
                offset=second_offset,
            ),
        )
        with self.assertRaisesRegex(cooker.CookError, "positive"):
            normalizer.normalized_mesh(parsed, 0.0, "y")

    def test_explicit_hull_cook_is_byte_deterministic(self) -> None:
        first = cooker.cook_asset(TETRA_OBJ, "fixture/tetra.obj", method="hull")
        second = cooker.cook_asset(TETRA_OBJ, "fixture/tetra.obj", method="hull")

        self.assertEqual(
            cooker.canonical_json_bytes(first), cooker.canonical_json_bytes(second)
        )
        self.assertEqual(len(first["parts"]), 1)
        self.assertEqual(first["parts"][0]["stableID"],
                         "hull-" + first["parts"][0]["digest"][:16])
        cooker.validate_asset(first)

    def test_connected_components_are_split_and_sorted(self) -> None:
        asset = cooker.cook_asset(
            two_tetra_obj(), "fixture/two-tetra.obj", method="hull"
        )
        self.assertEqual(len(asset["parts"]), 2)
        self.assertEqual(
            [part["digest"] for part in asset["parts"]],
            sorted(part["digest"] for part in asset["parts"]),
        )

    def test_scale_aware_weld_joins_duplicate_seam_vertices(self) -> None:
        # Two closed tetrahedra use distinct OBJ indices for their shared
        # origin. Geometric seam welding must discover one component even
        # though there is no indexed-topology edge between them.
        joined = TETRA_OBJ.decode("utf-8") + """\
v 0 0 0
v -1 0 0
v 0 -1 0
v 0 0 -1
f 5 7 6
f 5 6 8
f 5 8 7
f 6 7 8
"""
        mesh = cooker.parse_obj(joined.encode("utf-8"))
        components = cooker.connected_components(mesh, cooker.CookParameters())
        self.assertEqual(len(components), 1)

        scaled = cooker.scaled_mesh(mesh, (1000.0, 1000.0, 1000.0))
        components = cooker.connected_components(scaled, cooker.CookParameters())
        self.assertEqual(len(components), 1)

    def test_coacd_is_strictly_required_without_hull_opt_in(self) -> None:
        with mock.patch.object(
            cooker.importlib, "import_module", side_effect=ModuleNotFoundError("coacd")
        ):
            with self.assertRaisesRegex(cooker.CookError, "no implicit fallback"):
                cooker.cook_asset(TETRA_OBJ, "tetra.obj")

    def test_coacd_pins_seed_and_distinguishes_part_and_vertex_limits(self) -> None:
        fake = FakeCoACD()
        parameters = cooker.CookParameters(
            max_hulls=7,
            max_vertices_per_hull=23,
            seed=1729,
        )
        numpy_stub = types.SimpleNamespace(
            float64=object(),
            int32=object(),
            asarray=lambda values, dtype: values,
        )
        with mock.patch.dict(sys.modules, {"numpy": numpy_stub}):
            asset = cooker.cook_asset(
                TETRA_OBJ,
                "tetra.obj",
                method="coacd",
                parameters=parameters,
                coacd_backend=fake,
                coacd_backend_version=fake.__version__,
            )

        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["max_convex_hull"], 7)
        self.assertEqual(call["max_ch_vertex"], 23)
        self.assertEqual(call["seed"], 1729)
        self.assertTrue(call["real_metric"])
        self.assertTrue(call["merge"])
        self.assertTrue(call["decimate"])
        self.assertFalse(call["pca"])
        self.assertEqual(call["apx_mode"], "ch")
        self.assertEqual(call["extrude_margin"], 0.01)
        self.assertEqual(
            asset["cooker"]["backendVersion"], cooker.PINNED_COACD_VERSION
        )
        self.assertEqual(asset["cooker"]["parameters"]["maxHulls"], 7)
        self.assertEqual(
            asset["cooker"]["parameters"]["maxVerticesPerHull"], 23
        )
        with self.assertRaisesRegex(cooker.CookError, "1.0.11"):
            cooker.cook_asset(
                TETRA_OBJ,
                "tetra.obj",
                method="coacd",
                parameters=parameters,
                coacd_backend=fake,
                coacd_backend_version="1.0.10",
            )

    def test_cache_key_covers_source_scale_backend_seed_and_both_limits(self) -> None:
        base_parameters = cooker.CookParameters(max_hulls=8, max_vertices_per_hull=32)

        def key(
            *, source=TETRA_OBJ, scale=(1.0, 1.0, 1.0),
            parameters=base_parameters
        ) -> str:
            return cooker.cook_asset(
                source, "tetra.obj", method="hull", scale=scale,
                parameters=parameters
            )["cacheKey"]

        baseline = key()
        variants = [
            key(source=TETRA_OBJ.replace(b"v 1 0 0", b"v 2 0 0")),
            key(source=TETRA_OBJ + b"# same geometry, different source bytes\n"),
            key(scale=(2.0, 1.0, 1.0)),
            key(parameters=cooker.CookParameters(
                max_hulls=9, max_vertices_per_hull=32)),
            key(parameters=cooker.CookParameters(
                max_hulls=8, max_vertices_per_hull=31)),
            key(parameters=cooker.CookParameters(
                max_hulls=8, max_vertices_per_hull=32, seed=1)),
        ]
        self.assertTrue(all(value != baseline for value in variants))
        self.assertEqual(len(set(variants)), len(variants))
        uri_variant = cooker.cook_asset(
            TETRA_OBJ, "renamed/tetra.obj", method="hull",
            parameters=base_parameters,
        )["cacheKey"]
        self.assertNotEqual(uri_variant, baseline)

    def test_up_axis_is_baked_into_runtime_z_up_geometry(self) -> None:
        source = cooker.parse_obj(TETRA_OBJ, "tetra.obj")
        scale = (2.0, 3.0, 4.0)
        expected_vertices = {
            "z": {(0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                  (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)},
            "y": {(0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                  (0.0, 0.0, 3.0), (0.0, -4.0, 0.0)},
            "x": {(0.0, 0.0, 0.0), (0.0, 0.0, 2.0),
                  (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)},
        }
        assets = {}
        for axis in ("x", "y", "z"):
            with self.subTest(axis=axis):
                baked = cooker.baked_mesh(source, scale, axis)
                self.assertEqual(set(baked.vertices), expected_vertices[axis])
                asset = cooker.cook_asset(
                    TETRA_OBJ, "tetra.obj", method="hull",
                    scale=scale, up_axis=axis,
                )
                self.assertEqual(asset["source"]["upAxis"], axis)
                self.assertEqual(
                    asset["source"]["geometrySHA256"],
                    cooker.geometry_digest(baked),
                )
                assets[axis] = asset

        self.assertEqual(len({asset["cacheKey"] for asset in assets.values()}), 3)
        self.assertEqual(
            len({asset["source"]["geometrySHA256"] for asset in assets.values()}),
            3,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "tetra.obj"
            source_path.write_bytes(TETRA_OBJ)
            for axis, asset in assets.items():
                path = root / f"tetra-{axis}.avbdconvex.json"
                path.write_bytes(cooker.canonical_json_bytes(asset))
                cooker.verify_asset_file(path, source_path)

    def test_open_or_nonconvex_backend_output_is_rejected(self) -> None:
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        with self.assertRaisesRegex(
            cooker.CookError, "triangle count|open/non-manifold"
        ):
            cooker.canonicalize_hull(
                vertices,
                [(0, 2, 1), (0, 1, 3), (0, 3, 2)],
                max_vertices=64,
            )

        # A closed triangulated cube plus an interior-point-facing replacement
        # is topologically closed but geometrically concave.
        concave_vertices = vertices + [(0.15, 0.15, 0.15)]
        concave_triangles = [
            (0, 2, 1), (0, 1, 4), (1, 3, 4), (3, 0, 4),
            (0, 3, 2), (2, 3, 1),
        ]
        with self.assertRaises(cooker.CookError):
            cooker.canonicalize_hull(
                concave_vertices, concave_triangles, max_vertices=64
            )

    def test_obj_quads_must_be_planar_and_strictly_convex(self) -> None:
        header = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
"""
        valid = (header + "f 1 2 3 4\nf 1 5 2\nf 2 5 3\nf 3 5 4\nf 4 5 1\n").encode()
        mesh = cooker.parse_obj(valid, "valid-quad.obj")
        self.assertEqual(mesh.triangles[:2], ((0, 1, 2), (0, 2, 3)))

        nonplanar = valid.replace(b"v 0 1 0", b"v 0 1 0.1")
        with self.assertRaisesRegex(cooker.CookError, "non-planar quad"):
            cooker.parse_obj(nonplanar, "nonplanar.obj")

        concave = valid.replace(b"v 1 1 0", b"v 0.25 0.25 0")
        with self.assertRaisesRegex(cooker.CookError, "strictly convex"):
            cooker.parse_obj(concave, "concave.obj")

    def test_canonical_asset_and_debug_obj_are_atomically_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tetra.obj"
            output = root / "tetra.avbdconvex.json"
            debug = root / "tetra.debug.obj"
            cache = root / "cache"
            source.write_bytes(TETRA_OBJ)

            result = cooker.main([
                "--input", str(source),
                "--output", str(output),
                "--debug-obj", str(debug),
                "--cache-directory", str(cache),
                "--method", "hull",
            ])
            self.assertEqual(result, 0)
            asset = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(output.read_bytes(), cooker.canonical_json_bytes(asset))
            self.assertTrue(debug.read_text(encoding="utf-8").startswith(
                "# Deterministic AVBD convex-compound debug mesh\n"
            ))
            self.assertEqual(
                (cache / f"{asset['cacheKey']}.avbdconvex.json").read_bytes(),
                output.read_bytes(),
            )

            # Verification is hermetic: it reads canonical JSON and the
            # optional source, and never imports the decomposition backend.
            with mock.patch.object(
                cooker.importlib, "import_module",
                side_effect=AssertionError("verify tried to import CoACD"),
            ):
                self.assertEqual(cooker.main([
                    "--verify", str(output), "--input", str(source),
                    "--debug-obj", str(debug),
                ]), 0)

            for path in (output, debug, cache / f"{asset['cacheKey']}.avbdconvex.json"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

            debug.write_bytes(debug.read_bytes() + b"# tampered\n")
            with self.assertRaisesRegex(cooker.CookError, "debug OBJ"):
                cooker.verify_asset_file(output, source, debug)

            source.write_bytes(TETRA_OBJ + b"# changed\n")
            with self.assertRaisesRegex(cooker.CookError, "byte count|SHA-256"):
                cooker.verify_asset_file(output, source)

    def test_verify_rejects_noncanonical_or_unknown_json(self) -> None:
        asset = cooker.cook_asset(TETRA_OBJ, "tetra.obj", method="hull")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.json"
            path.write_text(json.dumps(asset, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(cooker.CookError, "not canonical"):
                cooker.verify_asset_file(path)

            asset["unexpected"] = True
            path.write_bytes(cooker.canonical_json_bytes(asset))
            with self.assertRaisesRegex(cooker.CookError, "keys differ"):
                cooker.verify_asset_file(path)

            with mock.patch("builtins.print"):
                self.assertEqual(cooker.main([
                    "--verify", str(path), "--scale", "2", "2", "2",
                ]), 2)
            with mock.patch.object(cooker.sys, "stderr"):
                with self.assertRaises(SystemExit):
                    cooker.main([
                        "--verify", str(path), "--sca", "2", "2", "2"
                    ])

    def test_untrusted_json_type_confusion_and_missing_fields_raise_cook_error(self) -> None:
        base = cooker.cook_asset(TETRA_OBJ, "tetra.obj", method="hull")

        def clone():
            return json.loads(cooker.canonical_json_bytes(base))

        malformed = []
        value = clone()
        value["schemaVersion"] = True
        malformed.append(value)
        value = clone()
        value["cooker"]["parameters"]["thresholdMeters"] = True
        malformed.append(value)
        value = clone()
        value["parts"][0]["triangles"][0][0] = False
        malformed.append(value)
        value = clone()
        value["parts"][0]["edges"][0]["vertexA"] = False
        malformed.append(value)
        value = clone()
        del value["parts"][0]["digest"]
        malformed.append(value)
        value = clone()
        value["parts"][0] = 7
        malformed.append(value)

        for index, asset in enumerate(malformed):
            with self.subTest(index=index):
                with self.assertRaises(cooker.CookError):
                    cooker.validate_asset(asset)

    def test_cache_seal_rejects_forged_source_or_cooker_identity(self) -> None:
        base = cooker.cook_asset(TETRA_OBJ, "tetra.obj", method="hull")

        def clone():
            return json.loads(cooker.canonical_json_bytes(base))

        for field, value in (
            ("sha256", "0" * 64),
            ("byteCount", len(TETRA_OBJ) + 1),
            ("uri", "different/tetra.obj"),
        ):
            asset = clone()
            asset["source"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(cooker.CookError, "cache key"):
                    cooker.validate_asset(asset)

        asset = clone()
        asset["cooker"]["backend"] = "coacd-python"
        with self.assertRaisesRegex(cooker.CookError, "identity"):
            cooker.validate_asset(asset)

        asset = clone()
        asset["cooker"]["algorithm"] = "coacd"
        asset["cooker"]["backend"] = "coacd-python"
        asset["cooker"]["backendVersion"] = "1.0.10"
        with self.assertRaisesRegex(cooker.CookError, "identity"):
            cooker.validate_asset(asset)

    def test_duplicate_parts_and_runtime_complexity_bounds_are_rejected(self) -> None:
        asset = cooker.cook_asset(
            two_tetra_obj(), "two-tetra.obj", method="hull"
        )
        asset["parts"] = [asset["parts"][0], asset["parts"][0]]
        asset["digest"] = cooker.compound_digest(asset["parts"])
        with self.assertRaisesRegex(cooker.CookError, "duplicate parts"):
            cooker.validate_asset(asset)

        with self.assertRaises(cooker.CookError):
            cooker.CookParameters(max_hulls=257).validate()
        with self.assertRaises(cooker.CookError):
            cooker.CookParameters(max_vertices_per_hull=65).validate()

        asset = cooker.cook_asset(TETRA_OBJ, "tetra.obj", method="hull")
        asset["parts"][0]["triangles"] = (
            asset["parts"][0]["triangles"] * 32
        )[: cooker.MAX_TRIANGLES_PER_HULL + 1]
        with self.assertRaisesRegex(cooker.CookError, "triangle count"):
            cooker.validate_asset(asset)

    def test_tiny_valid_hull_rejects_dimensionally_forged_volume(self) -> None:
        asset = cooker.cook_asset(TINY_TETRA_OBJ, "tiny.obj", method="hull")
        cooker.validate_asset(asset)
        self.assertLess(asset["parts"][0]["volume"], 2.0e-13)

        asset["parts"][0]["volume"] = 1.0e-8
        with self.assertRaisesRegex(cooker.CookError, "stored volume"):
            cooker.validate_asset(asset)

    def test_translated_tetra_uses_hull_local_volume_reference(self) -> None:
        asset = cooker.cook_asset(
            TRANSLATED_TETRA_OBJ, "translated.obj", method="hull"
        )
        cooker.validate_asset(asset)
        part = asset["parts"][0]
        self.assertAlmostEqual(part["volume"], 1.0 / 6.0, places=6)
        self.assertEqual(part["centroid"], [1000000.25, 1000000.25, 1000000.25])

    def test_merged_coplanar_face_loop_must_fit_gpu_workspace(self) -> None:
        prism_vertices, prism_triangles = regular_polygon_prism(17)
        with self.assertRaisesRegex(cooker.CookError, "17 vertices"):
            cooker.validate_merged_face_loops(prism_vertices, prism_triangles)

        # A 17-gon pyramid stays inside the whole-hull 64-vertex limit, proving
        # the independent face-loop limit is enforced by canonical cooking.
        base = prism_vertices[:17]
        pyramid_vertices = base + [(0.0, 0.0, 1.0)]
        pyramid_triangles = [
            (0, index + 1, index) for index in range(1, 16)
        ] + [
            (index, (index + 1) % 17, 17) for index in range(17)
        ]
        with self.assertRaisesRegex(cooker.CookError, "17 vertices"):
            cooker.canonicalize_hull(
                pyramid_vertices, pyramid_triangles, max_vertices=64
            )

    def test_asset_size_and_source_uri_are_bounded_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oversized = Path(temporary) / "oversized.json"
            with oversized.open("wb") as file:
                file.truncate(cooker.MAX_ASSET_BYTES + 1)
            with self.assertRaisesRegex(cooker.CookError, "maximum"):
                cooker.load_canonical_asset(oversized)

        with self.assertRaisesRegex(cooker.CookError, "source URI"):
            cooker.cook_asset(
                TETRA_OBJ,
                "a" * (cooker.MAX_SOURCE_URI_BYTES + 1),
                method="hull",
            )

    def test_multi_output_publish_rolls_back_all_originals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"old-first")
            second.write_bytes(b"old-second")
            real_replace = os.replace
            replacements = 0

            def fail_fourth_replace(source, destination):
                nonlocal replacements
                replacements += 1
                if replacements == 4:
                    raise OSError("injected publish failure")
                return real_replace(source, destination)

            with mock.patch.object(cooker.os, "replace", fail_fourth_replace):
                with self.assertRaisesRegex(OSError, "injected"):
                    cooker.atomic_write_many([
                        (first, b"new-first"), (second, b"new-second")
                    ])
            self.assertEqual(first.read_bytes(), b"old-first")
            self.assertEqual(second.read_bytes(), b"old-second")
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["first", "second"])

    def test_cli_never_overwrites_an_aliased_input_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            original = TETRA_OBJ
            source.write_bytes(original)

            with mock.patch("builtins.print"):
                self.assertEqual(cooker.main([
                    "--input", str(source), "--output", str(source),
                    "--method", "hull",
                ]), 2)
            self.assertEqual(source.read_bytes(), original)

            output = root / "asset.json"
            debug_alias = root / "debug.obj"
            debug_alias.symlink_to(source)
            with mock.patch("builtins.print"):
                self.assertEqual(cooker.main([
                    "--input", str(source), "--output", str(output),
                    "--debug-obj", str(debug_alias), "--method", "hull",
                ]), 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(output.exists())

            hardlink_alias = root / "hardlink.obj"
            os.link(source, hardlink_alias)
            with self.assertRaisesRegex(cooker.CookError, "aliases"):
                cooker.atomic_write_many(
                    [(hardlink_alias, b"replacement")],
                    protected_paths=(source,),
                )
            self.assertEqual(source.read_bytes(), original)

    def test_checked_concave_fixture_has_meaningful_quality(self) -> None:
        fixture_root = TOOLS.parent / "Sources/PhysicsAVBD/Assets/convex"
        asset = cooker.verify_asset_file(
            fixture_root / "concave-u.avbdconvex.json",
            fixture_root / "concave-u.obj",
            fixture_root / "concave-u.debug.obj",
        )
        source_mesh = cooker.parse_obj((fixture_root / "concave-u.obj").read_bytes())
        self.assertEqual(
            len(cooker.connected_components(source_mesh, cooker.CookParameters())), 1
        )
        self.assertEqual(len(asset["parts"]), 3)
        self.assertAlmostEqual(
            sum(part["volume"] for part in asset["parts"]), 7.0, delta=0.02
        )
        self.assertTrue(all(
            4 <= len(part["vertices"]) <= cooker.MAX_VERTICES_PER_HULL
            and 4 <= len(part["triangles"]) <= cooker.MAX_TRIANGLES_PER_HULL
            and 6 <= len(part["edges"]) <= cooker.MAX_EDGES_PER_HULL
            for part in asset["parts"]
        ))
        cavity_point = (0.0, 0.0, 0.75)
        for part in asset["parts"]:
            vertices = [tuple(vertex) for vertex in part["vertices"]]
            inside = all(
                cooker.dot(
                    cooker.triangle_normal(vertices, tuple(triangle)),
                    cooker.subtract(cavity_point, vertices[triangle[0]]),
                ) <= 1.0e-6
                for triangle in part["triangles"]
            )
            self.assertFalse(inside, "decomposition filled the authored U cavity")

    def test_checked_classic_fixtures_are_grounded_and_budgeted(self) -> None:
        asset_root = TOOLS.parent / "Sources/PhysicsAVBD/Assets"
        source_root = asset_root / "classic"
        convex_root = asset_root / "convex/classic"
        expected = {
            "stanford-bunny": {
                "parts": 21,
                "cap": 32,
                "threshold": 0.12,
                "preprocess": ("on", 80),
                "sourceSHA256": (
                    "e0f8157b25b0a876583b0205655471dbec6b6c47a8434c967387500818d8db1c"
                ),
                "geometrySHA256": (
                    "85823815aef140cba4079077b3d836eb552acbde10aa849a2b74cd8e62286667"
                ),
                "digest": (
                    "bacac8091e9e1eabea1183d9abfa2cfd298d93331d26bc8a8abc2d0287cd22a1"
                ),
                "assetSHA256": (
                    "ba14d9486477f261d5f36f24f9b421b12012ff696f1a87da6b12690c4c2c3c77"
                ),
                "debugSHA256": (
                    "ced184df11ee24214040743e0d1f2f6452949073c83a3e3d778c0ab9ca5f7814"
                ),
                "volume": 0.05160232787602581,
            },
            "stanford-dragon": {
                "parts": 39,
                "cap": 48,
                "threshold": 0.055,
                "preprocess": ("on", 100),
                "sourceSHA256": (
                    "a86bf3b22d299d5baae35493d2ff56794bd8c4518044c6a66c09757d1c674252"
                ),
                "geometrySHA256": (
                    "4149f1efb0e9f694ba73c77f988297155298f4fd85be2f3c2d0aba77ff494e2f"
                ),
                "digest": (
                    "eefb3d70331ae9b87ecedc367ec5e60ba9f647ddd040933152c9b0180fe5caa7"
                ),
                "assetSHA256": (
                    "a1df287ac6a7debecfac19487b678859db274430484515d86a4ac5044770ec31"
                ),
                "debugSHA256": (
                    "1bca90b7d0177506e278256dbb6a947ef126e7cd746f1d03af6d5f181fef7d1f"
                ),
                "volume": 0.06522370866605343,
            },
            "stanford-armadillo": {
                "parts": 21,
                "cap": 32,
                "threshold": 0.04,
                "preprocess": ("auto", 80),
                "sourceSHA256": (
                    "ce9866e4804c4ffee0fef5043aaf0d5f14780b6a5c13e948d8254e34771fde1a"
                ),
                "geometrySHA256": (
                    "0de8c05b4418d7f10c6c1fed728b754e294d41f7461195c65723c55bfb0f2cb3"
                ),
                "digest": (
                    "7e730459134ab092c5e9872c3e5a9e1404f7b3af681bc3e0a670ad16bf31ec48"
                ),
                "assetSHA256": (
                    "0a02f6acb74f17518f1fe73db4c0e8eb3849f509d95992935e0c71ac86f9eaa0"
                ),
                "debugSHA256": (
                    "38199c3279bfce1b51a1a7c633f6eb98d953b966193c1a758e006b532fc294c2"
                ),
                "volume": 0.02764676432707347,
            },
            "utah-teapot": {
                "parts": 14,
                "cap": 24,
                "threshold": 0.025,
                "preprocess": ("auto", 80),
                "sourceSHA256": (
                    "83591f7524e6626da5dd019aa85f76782cb1f07b601440f61224c24821f5c6b7"
                ),
                "geometrySHA256": (
                    "20b499947ac63e917605e8107990c9fb08449b2635fbbe5e5afda912fe21b8f1"
                ),
                "digest": (
                    "77461425636283ed3a21e51dbe329feda1f28ff9aa34984d7662126f3767a1ba"
                ),
                "assetSHA256": (
                    "c284d669a4fb8994839df4deef00b35d8a7db7fe42c1bc902733975e8816e8be"
                ),
                "debugSHA256": (
                    "22e59f788d80c3c494f8d725123331bc5a2266a823b5b3ebd35a2ec959c13147"
                ),
                "volume": 0.15818233663594583,
            },
        }
        total_parts = 0
        for name, fixture in expected.items():
            with self.subTest(asset=name):
                source_path = source_root / f"{name}.obj"
                asset_path = convex_root / f"{name}.avbdconvex.json"
                debug_path = convex_root / f"{name}.debug.obj"
                asset = cooker.verify_asset_file(
                    asset_path, source_path, debug_path
                )
                parameters = asset["cooker"]["parameters"]
                source = asset["source"]
                self.assertEqual(len(asset["parts"]), fixture["parts"])
                self.assertLess(len(asset["parts"]), fixture["cap"])
                self.assertEqual(parameters["maxHulls"], fixture["cap"])
                self.assertEqual(parameters["maxVerticesPerHull"], 24)
                self.assertAlmostEqual(
                    parameters["thresholdMeters"], fixture["threshold"], places=7
                )
                self.assertEqual(
                    (
                        parameters["coacdPreprocessMode"],
                        parameters["coacdPreprocessResolution"],
                    ),
                    fixture["preprocess"],
                )
                self.assertEqual(source["bakedScale"], [1.0, 1.0, 1.0])
                self.assertEqual(source["upAxis"], "z")
                self.assertEqual(source["uri"], f"classic/{name}.obj")
                self.assertEqual(source["sha256"], fixture["sourceSHA256"])
                self.assertEqual(
                    source["geometrySHA256"], fixture["geometrySHA256"]
                )
                self.assertEqual(asset["digest"], fixture["digest"])
                self.assertEqual(
                    hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                    fixture["assetSHA256"],
                )
                self.assertEqual(
                    hashlib.sha256(debug_path.read_bytes()).hexdigest(),
                    fixture["debugSHA256"],
                )
                self.assertLess(asset_path.stat().st_size, 256 * 1024)
                self.assertLess(debug_path.stat().st_size, 128 * 1024)

                mesh = cooker.parse_obj(source_path.read_bytes(), str(source_path))
                bounds_min = [
                    min(point[axis] for point in mesh.vertices)
                    for axis in range(3)
                ]
                bounds_max = [
                    max(point[axis] for point in mesh.vertices)
                    for axis in range(3)
                ]
                self.assertAlmostEqual(bounds_min[2], 0.0, places=6)
                self.assertAlmostEqual(bounds_min[0] + bounds_max[0], 0.0, places=6)
                self.assertAlmostEqual(bounds_min[1] + bounds_max[1], 0.0, places=6)
                self.assertGreater(bounds_max[2], 0.6)
                self.assertLess(bounds_max[2], 0.8)

                volume = sum(part["volume"] for part in asset["parts"])
                self.assertAlmostEqual(volume, fixture["volume"], places=12)
                self.assertTrue(math.isfinite(volume))
                self.assertGreater(volume, 0.0)
                compound_bounds_min = [
                    min(part["boundsMin"][axis] for part in asset["parts"])
                    for axis in range(3)
                ]
                compound_bounds_max = [
                    max(part["boundsMax"][axis] for part in asset["parts"])
                    for axis in range(3)
                ]
                center_of_mass = [
                    sum(
                        part["volume"] * part["centroid"][axis]
                        for part in asset["parts"]
                    ) / volume
                    for axis in range(3)
                ]
                for axis in range(3):
                    self.assertTrue(math.isfinite(center_of_mass[axis]))
                    self.assertLessEqual(
                        compound_bounds_min[axis], center_of_mass[axis]
                    )
                    self.assertGreaterEqual(
                        compound_bounds_max[axis], center_of_mass[axis]
                    )
                self.assertTrue(all(
                    4 <= len(part["vertices"]) <= 24
                    and 4 <= len(part["triangles"])
                    <= cooker.MAX_TRIANGLES_PER_HULL
                    and part["volume"] > 0.0
                    and part["boundingRadius"] > 0.0
                    for part in asset["parts"]
                ))
                total_parts += len(asset["parts"])

        self.assertEqual(total_parts, 95)
        self.assertLessEqual(total_parts, 96)


if __name__ == "__main__":
    unittest.main()
