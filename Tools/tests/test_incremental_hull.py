"""The dependency-free hull builder on inputs that used to fold."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

_COOKER = Path(__file__).resolve().parents[1] / "cook_convex_asset.py"
_spec = importlib.util.spec_from_file_location("cook_convex_asset_hull_test", _COOKER)
cook = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cook
_spec.loader.exec_module(cook)

# A ten-vertex cell with coplanar facets; the old builder emitted 18
# triangles where 16 close it, one edge on four faces, volume 14% high.
FOLDED_CELL = [[-0.358549953, 0.105620705, -0.325368136], [-0.342842251, 0.134404957, 0.311532468], [-0.328197598, 0.171735525, -0.335452288], [-0.312604427, 0.197021022, 0.301940084], [0.000801036, 0.046962656, 0.314099997], [0.008666527, 0.034110315, -0.325686872], [0.292155325, -0.184376508, 0.338228911], [0.305396199, -0.206217185, -0.300453633], [0.360051453, -0.14066568, 0.330219746], [0.37512368, -0.158595815, -0.309060276]]


def _edge_counts(triangles):
    counts = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _check(test, points):
    mesh = cook.incremental_convex_hull([tuple(p) for p in points])
    counts = _edge_counts(mesh.triangles)
    test.assertTrue(all(n == 2 for n in counts.values()), "every edge on exactly two faces")
    used = {index for triangle in mesh.triangles for index in triangle}
    test.assertEqual(len(mesh.triangles), 2 * len(used) - 4, "closed triangulated polytope")
    volume = cook.mesh_volume(mesh.vertices, mesh.triangles) if hasattr(cook, "mesh_volume") else None
    return mesh, volume


class IncrementalHullTests(unittest.TestCase):
    def test_folded_cell_is_a_closed_manifold(self):
        _check(self, FOLDED_CELL)

    def test_every_hull_of_a_cooked_arena(self):
        path = os.environ.get("CLATTER_ARENA_JSON")
        if not path:
            self.skipTest("set CLATTER_ARENA_JSON to check a whole document")
        document = json.load(open(path))
        for key, asset in document["hullAssets"].items():
            with self.subTest(hull=key):
                _check(self, [v[:3] for v in asset["vertices"]])


if __name__ == "__main__":
    unittest.main()
