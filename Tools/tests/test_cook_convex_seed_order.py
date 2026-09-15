"""Cooker half of the coplanar-face seeding fix, on the same captured
geometry the runtime tests use."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

_COOKER = Path(__file__).resolve().parents[1] / "cook_convex_asset.py"
_spec = importlib.util.spec_from_file_location("cook_convex_asset_seed_order", _COOKER)
cook = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cook
_spec.loader.exec_module(cook)

# One authored cell of a generated kitchen arena, refused before the fix with
# "coplanar face 60 boundary is disconnected". Float32 vertices about the centroid.
RESCUED_CELL = [[-0.048016801,1.041641474,-0.230060354],[-0.053028058,-0.314696729,-0.192076474],[0.052417245,1.067343116,0.700950027],[0.047405992,-0.288995087,0.738933921],[-0.047695171,1.041524649,-0.234189391],[-0.052706428,-0.314813524,-0.196205512],[-0.052309521,-0.318540931,-0.192287639],[0.048150077,-0.292832762,0.738959551],[-0.051523756,-0.319150388,-0.19709678],[-0.045957107,1.04140234,-0.238326803],[-0.050968364,-0.314935833,-0.200342923],[-0.050234977,-0.32182312,-0.193103656],[0.050297376,-0.296096325,0.738818049],[-0.04840818,-0.319306195,-0.202249095],[-0.048030302,-0.322854877,-0.198519036],[-0.043067228,1.041293144,-0.241842732],[-0.048078485,-0.315044969,-0.203858852],[-0.047120266,-0.324043632,-0.194400281],[0.053520981,-0.298289001,0.738530815],[-0.039465476,1.04121387,-0.244201884],[-0.044476733,-0.315124333,-0.206218004],[-0.044409126,-0.31893599,-0.205352709],[-0.04418828,-0.322125494,-0.203103215],[-0.043847818,-0.324207276,-0.19981201],[-0.043439567,-0.324864417,-0.195980117],[0.057330128,-0.299076915,0.738141656],[-0.000963504,1.040602922,-0.260938346],[-0.00597476,-0.315735281,-0.222954467],[-0.005907155,-0.319546938,-0.222089171],[-0.005686309,-0.322736442,-0.219839692],[-0.005345844,-0.324818224,-0.216548473],[-0.004937594,-0.325475365,-0.212716594],[0.097174682,-0.299344301,0.733850718],[0.002801776,1.040565372,-0.261781603],[-0.00220948,-0.315772831,-0.223797709],[-0.001251262,-0.324771494,-0.214339137],[0.100989461,-0.298607528,0.733418882],[-0.001632785,-0.320048422,-0.222581938],[-0.001254906,-0.323597103,-0.218851894],[0.006157361,1.040575385,-0.260980517],[0.001146105,-0.315762788,-0.222996637],[0.001879493,-0.322650075,-0.215757355],[0.104229107,-0.296458244,0.733010054],[0.00226551,-0.320003927,-0.22047852],[0.008592394,1.040631413,-0.258657098],[0.003581138,-0.3157067,-0.220673218],[0.003978045,-0.319434106,-0.216755345],[0.106400415,-0.293223649,0.73268652],[0.009736157,1.040724993,-0.25516507],[0.004724901,-0.315613151,-0.217181176],[0.112184078,1.066941977,0.694513679],[0.107172824,-0.289396167,0.732497573]]

# One cell with a sub-Float32 step: a 2 um triangle 38 nm off collinear on a
# 5 mm part. It must stay refused as degenerate rather than be admitted.
DEGENERATE_CELL = [[-0.001304534,-0.001253359,0.000117088],[0.001295448,-0.001262405,0.000120731],[-0.001450428,-0.001239952,0.000129771],[0.001441395,-0.001250014,0.000133823],[-0.001590667,-0.001203948,0.00016506],[0.001581782,-0.001214987,0.000169505],[-0.001726945,-0.001144683,0.00022361],[0.001718305,-0.00115667,0.000228437],[-0.001842053,-0.001062733,0.000304928],[0.001833751,-0.001075523,0.000310078],[-0.001925849,-0.00096532,0.000401849],[0.001917951,-0.000978694,0.000407235],[-0.001950547,-0.000915313,0.000451692],[0.001942857,-0.00092886,0.000457148],[-0.00197445,-0.000861773,0.00050507],[0.001966983,-0.000875487,0.000510592],[-0.001990187,-0.000822281,0.000544451],[0.001982884,-0.000836105,0.000550018],[-0.002003437,-0.000725629,0.000640956],[0.001996535,-0.000739546,0.00064656],[-0.002001957,-1.3831e-05,0.00135215],[0.001998015,-2.7748e-05,0.001357755],[-0.001944457,0.000554664,-0.000244323],[0.001948947,0.000541117,-0.000238868],[-0.001984098,0.000647695,-0.000151565],[0.001988974,0.000633872,-0.000145998],[-0.001997347,0.000744348,-5.506e-05],[0.002002625,0.00073043,-4.9455e-05],[-0.001297901,0.000347656,-0.000640972],[0.00130208,0.000338609,-0.00063733],[-0.001443795,0.000361062,-0.000628289],[0.001448028,0.000351001,-0.000624237],[-0.001584034,0.000397066,-0.000593],[0.001588414,0.000386028,-0.000588555],[-0.001720313,0.000456332,-0.00053445],[0.001724937,0.000444344,-0.000529623],[-0.00183542,0.000538281,-0.000453133],[0.001840384,0.000525491,-0.000447982],[-0.001919216,0.000635694,-0.000356211],[0.001924584,0.00062232,-0.000350826],[-0.001920809,0.00063892,-0.000352997],[0.00192619,0.000625535,-0.000347606],[-0.001966953,0.000737304,-0.000254922],[0.001972743,0.000723596,-0.000249402],[-0.001967818,0.000739241,-0.000252991],[0.001973615,0.000725527,-0.000247468],[-0.001968387,0.00074067,-0.000251566],[0.001974191,0.000726952,-0.000246042],[-0.001983532,0.000789854,-0.000202498],[0.00198954,0.00077603,-0.000196931],[-0.001996781,0.000886506,-0.000105993],[0.002003191,0.000872589,-0.000100389],[-0.001996202,0.001165021,0.000172285],[0.00200377,0.001151104,0.00017789]]


def _hull(points):
    mesh = cook.incremental_convex_hull([tuple(p) for p in points])
    return list(mesh.vertices), list(mesh.triangles)


def _scipy_hull(points):
    from scipy.spatial import ConvexHull  # optional; the offline cooker never needs it

    hull = ConvexHull(points)
    return [tuple(map(float, p)) for p in points], [tuple(map(int, t)) for t in hull.simplices]


def _prism(sides):
    points = []
    for i in range(sides):
        angle = 2.0 * math.pi * i / sides
        points.append((math.cos(angle), math.sin(angle), 0.0))
        points.append((math.cos(angle), math.sin(angle), 0.2))
    return points


class SeedOrderTests(unittest.TestCase):
    def test_rescued_cell_canonicalises(self):
        try:
            vertices, triangles = _scipy_hull(RESCUED_CELL)
        except ImportError:
            self.skipTest("scipy not available for the captured triangulation")
        asset = cook.canonicalize_hull(vertices, triangles, 64)
        self.assertGreaterEqual(len(asset["vertices"]), 4)

    def test_sub_resolution_cell_is_still_refused_as_degenerate(self):
        try:
            vertices, triangles = _scipy_hull(DEGENERATE_CELL)
        except ImportError:
            self.skipTest("scipy not available for the captured triangulation")
        with self.assertRaises(cook.CookError) as raised:
            cook.canonicalize_hull(vertices, triangles, 64)
        self.assertIn("degenerate", str(raised.exception))

    def test_merged_face_limit_is_evaluated_on_merged_faces(self):
        # At the 64-vertex hull cap a face loop cannot exceed 64 vertices at
        # all (it is a subset of the hull's vertices); at cap 256 the limit is
        # what decides, on the merged face.
        for sides, expected_ok in ((60, True), (64, True), (65, False), (70, False)):
            vertices, triangles = _hull(_prism(sides))
            with self.subTest(sides=sides):
                if expected_ok:
                    cook.canonicalize_hull(vertices, triangles, 256)
                else:
                    with self.assertRaises(cook.CookError) as raised:
                        cook.canonicalize_hull(vertices, triangles, 256)
                    self.assertIn("runtime supports at most", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
