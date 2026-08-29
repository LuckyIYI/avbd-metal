#!/usr/bin/env python3

from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
