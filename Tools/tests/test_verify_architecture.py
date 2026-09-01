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


def product_dependency(name: str, package: str) -> dict[str, list[object]]:
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


def valid_simulator_manifest() -> dict[str, object]:
    return {
        "name": "gpu-sim",
        "dependencies": [],
        "targets": [
            target("SimCore"),
            target(
                "PhysicsAVBD", [target_dependency("SimCore")],
                resources=[resource("Shaders")],
            ),
            target(
                "GPUSim",
                [target_dependency("SimCore"), target_dependency("PhysicsAVBD")],
            ),
            target(
                "GPUSimDemos", [target_dependency("SimCore")],
                resources=[resource("Assets")],
            ),
            target(
                "GPUSimRenderer",
                [target_dependency("SimCore"), target_dependency("PhysicsAVBD")],
            ),
            target("SimCoreTests", target_type="test"),
        ],
        "products": [product(name) for name in architecture.SIMULATOR_TARGETS],
    }


def valid_development_manifest() -> dict[str, object]:
    core = product_dependency("SimCore", "gpu-sim")
    physics = product_dependency("PhysicsAVBD", "gpu-sim")
    demos = product_dependency("GPUSimDemos", "gpu-sim")
    renderer = product_dependency("GPUSimRenderer", "gpu-sim")
    mlx = [product_dependency(name, "mlx-swift")
           for name in architecture.MLX_PRODUCTS]
    return {
        "name": "avbd-metal",
        "targets": [
            target("Robotics", [core], resources=[resource("Assets")]),
            target(
                "RL", [core, physics, demos, target_dependency("Robotics")]
            ),
            target(
                "MLXRL",
                [core, physics, target_dependency("Robotics"),
                 target_dependency("RL"), *mlx],
            ),
            target(
                "avbd",
                [core, physics, demos, target_dependency("Robotics"),
                 target_dependency("RL"), target_dependency("MLXRL")],
                target_type="executable",
            ),
            target(
                "AVBDApp",
                [core, physics, demos, renderer, target_dependency("Robotics"),
                 target_dependency("RL"), target_dependency("MLXRL")],
                target_type="executable",
            ),
            target("AVBDTests", target_type="test"),
        ],
        "products": [
            *(product(name) for name in architecture.DEVELOPMENT_LAYER_TARGETS),
            *(product(name, "executable") for name in architecture.ENTRY_POINT_TARGETS),
        ],
    }


def find_target(manifest: dict[str, object], name: str) -> dict[str, object]:
    targets = manifest["targets"]
    assert isinstance(targets, list)
    return next(item for item in targets if item["name"] == name)


def write(root: Path, relative_path: str, contents: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def create_valid_sources(root: Path) -> None:
    write(root, "Sources/SimCore/Core.swift", "public struct Scene {}\n")
    write(root, "Sources/PhysicsAVBD/Physics.swift", "import SimCore\n")
    write(
        root, "Sources/GPUSim/GPUSim.swift",
        "@_exported import SimCore\n@_exported import PhysicsAVBD\n",
    )
    write(root, "Sources/GPUSimDemos/Demos.swift",
          "import SimCore\npublic enum Demos {}\n")
    write(
        root, "Sources/GPUSimRenderer/Renderer.swift",
        "import SimCore\nimport PhysicsAVBD\npublic struct Renderer {}\n",
    )
    write(root, "Sources/PhysicsAVBD/Shaders/solver.metal",
          "kernel void solve() {}\n")
    write(root, "Sources/GPUSimDemos/Assets/bunny.obj", "o bunny\n")

    development = root / "Development"
    write(development, "Sources/Robotics/Robot.swift", "import SimCore\n")
    write(
        development, "Sources/RL/Environment.swift",
        "import SimCore\nimport PhysicsAVBD\nimport GPUSimDemos\nimport Robotics\n",
    )
    write(
        development, "Sources/MLXRL/Pipeline.swift",
        "\n".join([
            "import SimCore", "import PhysicsAVBD", "import Robotics", "import RL",
            "import MLX", "import MLXNN", "import MLXOptimizers",
            "import MLXRandom", "import MLXLinalg", "",
        ]),
    )
    imports = "\n".join(
        f"import {name}" for name in
        ("SimCore", "PhysicsAVBD", "GPUSimDemos", "Robotics", "RL", "MLXRL")
    ) + "\n"
    write(development, "Sources/avbd/main.swift", imports)
    write(
        development,
        "Sources/AVBDApp/App.swift",
        imports + "import GPUSimRenderer\n",
    )
    write(development, "Sources/Robotics/Assets/robot.xml", "<mujoco/>\n")


class ManifestVerificationTests(unittest.TestCase):
    def test_standalone_simulator_manifest_passes(self) -> None:
        architecture.verify_manifest(valid_simulator_manifest())

    def test_development_manifest_passes(self) -> None:
        architecture.verify_manifest(
            valid_development_manifest(), architecture.DEVELOPMENT_CONTRACT
        )

    def test_root_backend_cannot_depend_on_demos(self) -> None:
        manifest = valid_simulator_manifest()
        physics = find_target(manifest, "PhysicsAVBD")
        dependencies = physics["dependencies"]
        assert isinstance(dependencies, list)
        dependencies.append(target_dependency("GPUSimDemos"))
        with self.assertRaisesRegex(
            architecture.VerificationError, r"target PhysicsAVBD dependencies"
        ):
            architecture.verify_manifest(manifest)

    def test_root_backend_cannot_own_demo_assets(self) -> None:
        manifest = valid_simulator_manifest()
        physics = find_target(manifest, "PhysicsAVBD")
        resources = physics["resources"]
        assert isinstance(resources, list)
        resources.append(resource("Assets"))
        with self.assertRaisesRegex(
            architecture.VerificationError, r"target PhysicsAVBD resources"
        ):
            architecture.verify_manifest(manifest)

    def test_facade_requires_both_neutral_scene_and_backend(self) -> None:
        manifest = valid_simulator_manifest()
        facade = find_target(manifest, "GPUSim")
        facade["dependencies"] = [target_dependency("SimCore")]
        with self.assertRaisesRegex(
            architecture.VerificationError, r"target GPUSim dependencies"
        ):
            architecture.verify_manifest(manifest)

    def test_development_reverse_dependency_is_rejected(self) -> None:
        manifest = valid_development_manifest()
        robotics = find_target(manifest, "Robotics")
        dependencies = robotics["dependencies"]
        assert isinstance(dependencies, list)
        dependencies.append(target_dependency("RL"))
        with self.assertRaisesRegex(
            architecture.VerificationError, r"target Robotics dependencies"
        ):
            architecture.verify_manifest(
                manifest, architecture.DEVELOPMENT_CONTRACT
            )

    def test_mlx_cannot_enter_root_manifest(self) -> None:
        root = valid_simulator_manifest()
        development = valid_development_manifest()
        root["dependencies"] = [{"sourceControl": [{}]}]
        development["dependencies"] = []
        with self.assertRaisesRegex(
            architecture.VerificationError, r"root GPUSim package must have zero"
        ):
            architecture.verify_package_dependencies(root, development, Path("."))

    def test_exact_local_and_mlx_package_dependencies_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root_path = Path(directory).resolve()
            simulator = valid_simulator_manifest()
            development = valid_development_manifest()
            development["dependencies"] = [
                {"fileSystem": [{
                    "nameForTargetDependencyResolutionOnly": "gpu-sim",
                    "path": str(root_path),
                }]},
                {"sourceControl": [{
                    "location": {"remote": [{"urlString":
                        "https://github.com/ml-explore/mlx-swift"}]}
                }]},
            ]
            architecture.verify_package_dependencies(
                simulator, development, root_path
            )

    def test_unknown_production_target_is_rejected(self) -> None:
        manifest = valid_simulator_manifest()
        targets = manifest["targets"]
        assert isinstance(targets, list)
        targets.append(target("LearningUtilities"))
        with self.assertRaisesRegex(
            architecture.VerificationError, r"unexpected=\[LearningUtilities\]"
        ):
            architecture.verify_manifest(manifest)

    def test_explicit_sources_cannot_hide_code(self) -> None:
        manifest = valid_simulator_manifest()
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

    def test_reexport_outside_facade_is_rejected(self) -> None:
        write(
            self.root, "Sources/PhysicsAVBD/Reexport.swift",
            "@_exported import SimCore\n",
        )
        with self.assertRaisesRegex(
            architecture.VerificationError, r"only GPUSim may re-export"
        ):
            architecture.verify_source_imports(self.root)

    def test_facade_must_reexport_both_modules(self) -> None:
        write(
            self.root, "Sources/GPUSim/GPUSim.swift",
            "@_exported import SimCore\nimport PhysicsAVBD\n",
        )
        with self.assertRaisesRegex(
            architecture.VerificationError, r"re-export exactly"
        ):
            architecture.verify_source_imports(self.root)

    def test_public_import_cannot_replace_facade_reexport(self) -> None:
        write(
            self.root, "Sources/GPUSim/GPUSim.swift",
            "public import SimCore\n@_exported import PhysicsAVBD\n",
        )
        with self.assertRaisesRegex(
            architecture.VerificationError, r"single-import contract"
        ):
            architecture.verify_source_imports(self.root)

    def test_backend_cannot_import_development_layer(self) -> None:
        write(
            self.root, "Sources/PhysicsAVBD/Reverse.swift",
            "import SimCore\nimport RL\n",
        )
        with self.assertRaisesRegex(
            architecture.VerificationError, r"PhysicsAVBD.*may not import RL"
        ):
            architecture.verify_source_imports(self.root)

    def test_mlx_import_below_mlxrl_is_rejected(self) -> None:
        write(self.root / "Development", "Sources/RL/Leak.swift", "import MLX\n")
        with self.assertRaisesRegex(
            architecture.VerificationError, r"MLX import MLX belongs in MLXRL"
        ):
            architecture.verify_source_imports(self.root)

    def test_production_testable_import_is_rejected(self) -> None:
        write(
            self.root / "Development", "Sources/RL/Testable.swift",
            "@testable import Robotics\n",
        )
        with self.assertRaisesRegex(
            architecture.VerificationError, r"production source cannot use @testable"
        ):
            architecture.verify_source_imports(self.root)

    def test_misplaced_demo_declaration_is_rejected(self) -> None:
        write(self.root, "Sources/SimCore/DemosExtra.swift", "extension Demos {}\n")
        with self.assertRaisesRegex(
            architecture.VerificationError, r"demos belong in Sources/GPUSimDemos"
        ):
            architecture.verify_source_imports(self.root)

    def test_unlisted_source_target_directory_is_rejected(self) -> None:
        (self.root / "Sources/RL").mkdir()
        with self.assertRaisesRegex(
            architecture.VerificationError, r"source target directories"
        ):
            architecture.verify_source_imports(self.root)

    def test_misplaced_metal_file_is_rejected(self) -> None:
        write(self.root, "Sources/SimCore/Shaders/render.metal",
              "kernel void render() {}\n")
        with self.assertRaisesRegex(
            architecture.VerificationError, r"shaders belong to PhysicsAVBD"
        ):
            architecture.verify_resource_layout(self.root)

    def test_misplaced_asset_is_rejected(self) -> None:
        write(self.root / "Development", "Sources/RL/Assets/robot.xml", "<x/>\n")
        with self.assertRaisesRegex(
            architecture.VerificationError, r"assets belong to GPUSimDemos or Robotics"
        ):
            architecture.verify_resource_layout(self.root)

    def test_external_arachne_project_is_rejected(self) -> None:
        (self.root / "Robots/Arachne15").mkdir(parents=True)
        with self.assertRaisesRegex(
            architecture.VerificationError, r"external project directory"
        ):
            architecture.verify_external_project_layout(self.root)


if __name__ == "__main__":
    unittest.main()
