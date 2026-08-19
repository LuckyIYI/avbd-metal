#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))

import verify_architecture as architecture


def target_dependency(name: str) -> dict[str, list[object]]:
    return {"byName": [name, None]}


def product_dependency(name: str, package: str = "mlx-swift") -> dict[str, list[object]]:
    return {"product": [name, package, None, None]}


def target(
    name: str,
    dependencies: list[dict[str, list[object]]] | None = None,
    *,
    target_type: str = "regular",
    resources: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "type": target_type,
        "dependencies": dependencies or [],
        "resources": resources or [],
        "exclude": [],
    }


def resource(path: str, rule: str = "copy") -> dict[str, object]:
    return {"path": path, "rule": {rule: {}}}


def product(name: str, kind: str = "library") -> dict[str, object]:
    return {
        "name": name,
        "targets": [name],
        "type": {kind: ["automatic"] if kind == "library" else None},
    }


def valid_manifest() -> dict[str, object]:
    mlx_dependencies = [
        product_dependency("MLX"),
        product_dependency("MLXNN"),
        product_dependency("MLXOptimizers"),
        product_dependency("MLXRandom"),
        product_dependency("MLXLinalg"),
    ]
    targets = [
        target("SimCore"),
        target(
            "PhysicsAVBD",
            [target_dependency("SimCore")],
            resources=[resource("Shaders")],
        ),
        target(
            "Robotics",
            [target_dependency("SimCore")],
            resources=[resource("Assets")],
        ),
        target(
            "RL",
            [
                target_dependency("SimCore"),
                target_dependency("PhysicsAVBD"),
                target_dependency("Robotics"),
            ],
        ),
        target(
            "MLXRL",
            [
                target_dependency("SimCore"),
                target_dependency("PhysicsAVBD"),
                target_dependency("Robotics"),
                target_dependency("RL"),
                *mlx_dependencies,
            ],
        ),
        target(
            "avbd",
            [target_dependency(name) for name in architecture.LAYER_TARGETS],
            target_type="executable",
        ),
        target(
            "AVBDApp",
            [target_dependency(name) for name in architecture.LAYER_TARGETS],
            target_type="executable",
        ),
        target("SimCoreTests", [target_dependency("SimCore")], target_type="test"),
    ]
    products = [
        *(product(name) for name in ("SimCore", "PhysicsAVBD", "Robotics", "RL", "MLXRL")),
        product("avbd", "executable"),
        product("AVBDApp", "executable"),
    ]
    return {"name": "fixture", "targets": targets, "products": products}


def find_target(manifest: dict[str, object], name: str) -> dict[str, object]:
    targets = manifest["targets"]
    assert isinstance(targets, list)
    return next(item for item in targets if item["name"] == name)


def write(root: Path, relative_path: str, contents: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def create_valid_sources(root: Path) -> None:
    write(
        root,
        "Sources/SimCore/Core.swift",
        '/* import RL\n nested /* import MLX */ comment */\nlet url = "https://example.test"\n',
    )
    write(root, "Sources/PhysicsAVBD/Physics.swift", "import SimCore\n")
    write(root, "Sources/PhysicsAVBD/Demos/Demos.swift", "public enum Demos {}\n")
    write(root, "Sources/Robotics/Robot.swift", "import SimCore\n")
    write(
        root,
        "Sources/RL/Environment.swift",
        "import SimCore\nimport PhysicsAVBD\nimport Robotics\n",
    )
    write(
        root,
        "Sources/MLXRL/Pipeline.swift",
        "\n".join(
            [
                "import SimCore",
                "import PhysicsAVBD",
                "import Robotics",
                "import RL",
                "import MLX",
                "import MLXNN",
                "import MLXOptimizers",
                "import MLXRandom",
                "import MLXLinalg",
                "",
            ]
        ),
    )
    write(
        root,
        "Sources/avbd/main.swift",
        "\n".join(f"import {name}" for name in architecture.LAYER_TARGETS) + "\n",
    )
    write(
        root,
        "Sources/AVBDApp/App.swift",
        "\n".join(f"import {name}" for name in architecture.LAYER_TARGETS) + "\n",
    )
    write(root, "Sources/PhysicsAVBD/Shaders/solver.metal", "kernel void solve() {}\n")
    write(root, "Sources/Robotics/Assets/robot.xml", "<mujoco/>\n")


class ManifestVerificationTests(unittest.TestCase):
    def test_exact_five_layer_manifest_passes(self) -> None:
        architecture.verify_manifest(valid_manifest())

    def test_reverse_dependency_is_rejected(self) -> None:
        manifest = valid_manifest()
        robotics = find_target(manifest, "Robotics")
        dependencies = robotics["dependencies"]
        assert isinstance(dependencies, list)
        dependencies.append(target_dependency("RL"))

        with self.assertRaisesRegex(
            architecture.VerificationError, r"target Robotics dependencies"
        ):
            architecture.verify_manifest(manifest)

    def test_rl_direct_physics_edge_is_required(self) -> None:
        manifest = valid_manifest()
        rl = find_target(manifest, "RL")
        dependencies = rl["dependencies"]
        assert isinstance(dependencies, list)
        rl["dependencies"] = [
            dependency
            for dependency in dependencies
            if dependency.get("byName", [None])[0] != "PhysicsAVBD"
        ]

        with self.assertRaisesRegex(
            architecture.VerificationError, r"target RL dependencies"
        ):
            architecture.verify_manifest(manifest)

    def test_mlxrl_requires_all_five_mlx_products_from_mlx_swift(self) -> None:
        manifest = valid_manifest()
        mlxrl = find_target(manifest, "MLXRL")
        dependencies = mlxrl["dependencies"]
        assert isinstance(dependencies, list)
        for dependency in dependencies:
            if dependency.get("product", [None])[0] == "MLXLinalg":
                dependency["product"][1] = "wrong-package"

        with self.assertRaisesRegex(
            architecture.VerificationError, r"target MLXRL dependencies"
        ):
            architecture.verify_manifest(manifest)

    def test_wrong_resource_owner_is_rejected_in_manifest(self) -> None:
        manifest = valid_manifest()
        robotics = find_target(manifest, "Robotics")
        robotics["resources"] = []
        physics = find_target(manifest, "PhysicsAVBD")
        resources = physics["resources"]
        assert isinstance(resources, list)
        resources.append(resource("Assets"))

        with self.assertRaisesRegex(
            architecture.VerificationError, r"target PhysicsAVBD resources"
        ):
            architecture.verify_manifest(manifest)

    def test_superseded_target_name_is_rejected(self) -> None:
        manifest = valid_manifest()
        targets = manifest["targets"]
        assert isinstance(targets, list)
        targets.append(target("PolicyRuntime"))

        with self.assertRaisesRegex(
            architecture.VerificationError, r"superseded target/product names.*PolicyRuntime"
        ):
            architecture.verify_manifest(manifest)

    def test_former_umbrella_target_name_is_rejected(self) -> None:
        manifest = valid_manifest()
        targets = manifest["targets"]
        assert isinstance(targets, list)
        targets.append(target("AVBDCore"))

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"superseded target/product names.*AVBDCore",
        ):
            architecture.verify_manifest(manifest)

    def test_former_umbrella_product_name_is_rejected(self) -> None:
        manifest = valid_manifest()
        products = manifest["products"]
        assert isinstance(products, list)
        products.append(product("AVBDLearn"))

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"superseded target/product names.*AVBDLearn",
        ):
            architecture.verify_manifest(manifest)

    def test_unknown_production_target_cannot_bypass_the_gate(self) -> None:
        manifest = valid_manifest()
        targets = manifest["targets"]
        assert isinstance(targets, list)
        targets.append(target("LearningUtilities"))

        with self.assertRaisesRegex(
            architecture.VerificationError, r"unexpected=\[LearningUtilities\]"
        ):
            architecture.verify_manifest(manifest)

    def test_explicit_sources_cannot_hide_unchecked_code(self) -> None:
        manifest = valid_manifest()
        find_target(manifest, "SimCore")["sources"] = ["Core.swift"]

        with self.assertRaisesRegex(
            architecture.VerificationError, r"explicit sources list"
        ):
            architecture.verify_manifest(manifest)


class SourceAndResourceVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        create_valid_sources(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_sources_and_resources_pass(self) -> None:
        architecture.verify_source_imports(self.root)
        architecture.verify_resource_layout(self.root)
        architecture.verify_external_project_layout(self.root)

    def test_reverse_import_is_rejected(self) -> None:
        write(
            self.root,
            "Sources/Robotics/Reverse.swift",
            "import SimCore\nimport RL\n",
        )

        with self.assertRaisesRegex(
            architecture.VerificationError, r"Robotics/Reverse.swift:2:.*may not import RL"
        ):
            architecture.verify_source_imports(self.root)

    def test_mlx_import_below_mlxrl_is_rejected(self) -> None:
        write(self.root, "Sources/RL/Leak.swift", "import MLX\n")

        with self.assertRaisesRegex(
            architecture.VerificationError, r"MLX import MLX belongs in MLXRL"
        ):
            architecture.verify_source_imports(self.root)

    def test_entry_point_cannot_bypass_mlxrl(self) -> None:
        write(
            self.root,
            "Sources/AVBDApp/App.swift",
            "\n".join(f"import {name}" for name in architecture.LAYER_TARGETS)
            + "\nimport MLX\n",
        )

        with self.assertRaisesRegex(
            architecture.VerificationError, r"MLX import MLX belongs in MLXRL"
        ):
            architecture.verify_source_imports(self.root)

    def test_production_testable_import_is_rejected(self) -> None:
        write(self.root, "Sources/RL/Testable.swift", "@testable import Robotics\n")

        with self.assertRaisesRegex(
            architecture.VerificationError, r"production source cannot use @testable"
        ):
            architecture.verify_source_imports(self.root)

    def test_former_umbrella_import_is_rejected(self) -> None:
        write(
            self.root,
            "Sources/MLXRL/CompatibilityLeak.swift",
            "import AVBDLearn\n",
        )

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"imports superseded module AVBDLearn",
        ):
            architecture.verify_source_imports(self.root)

    def test_production_reexport_is_rejected(self) -> None:
        write(
            self.root,
            "Sources/MLXRL/Exports.swift",
            "@_exported import RL\n",
        )

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"production modules may not re-export modules",
        ):
            architecture.verify_source_imports(self.root)

    def test_superseded_source_directory_is_rejected_even_when_empty(self) -> None:
        (self.root / "Sources" / "RLCore").mkdir()

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"superseded source directory remains: Sources/RLCore",
        ):
            architecture.verify_source_imports(self.root)

    def test_former_umbrella_source_directory_is_rejected_even_when_empty(self) -> None:
        (self.root / "Sources" / "AVBDCore").mkdir()

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"superseded source directory remains: Sources/AVBDCore",
        ):
            architecture.verify_source_imports(self.root)

    def test_avbd_tuned_demo_cannot_return_to_simcore(self) -> None:
        write(
            self.root,
            "Sources/SimCore/DemosExtra.swift",
            "extension Demos {}\n",
        )

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"SimCore/DemosExtra.swift: AVBD-tuned demos belong in",
        ):
            architecture.verify_source_imports(self.root)

    def test_misplaced_metal_file_is_rejected(self) -> None:
        write(self.root, "Sources/SimCore/Shaders/render.metal", "kernel void render() {}\n")

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"SimCore/Shaders/render.metal: shader resources belong to",
        ):
            architecture.verify_resource_layout(self.root)

    def test_misplaced_asset_is_rejected(self) -> None:
        write(self.root, "Sources/RL/Assets/robot.xml", "<mujoco/>\n")

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"RL/Assets/robot.xml: simulator/robot assets belong to",
        ):
            architecture.verify_resource_layout(self.root)

    def test_external_arachne_project_is_rejected(self) -> None:
        (self.root / "Robots" / "Arachne15").mkdir(parents=True)

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"external project directory must not be vendored: Robots/Arachne15",
        ):
            architecture.verify_external_project_layout(self.root)

    def test_dangling_external_project_symlink_is_rejected(self) -> None:
        robots = self.root / "Robots"
        robots.mkdir()
        (robots / "Arachne15").symlink_to(self.root / "missing-project")

        with self.assertRaisesRegex(
            architecture.VerificationError,
            r"external project directory must not be vendored: Robots/Arachne15",
        ):
            architecture.verify_external_project_layout(self.root)


if __name__ == "__main__":
    unittest.main()
