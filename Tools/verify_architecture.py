#!/usr/bin/env python3
"""Verify the standalone simulator and development-package boundaries.

The root Swift package must remain dependency-free and distributable as
``GPUSim``.  Robotics, RL, MLX, and application code live in the nested
``Development`` package and may depend on the root package, never the reverse.
The verifier uses only the Python standard library and evaluates manifests in
isolated temporary SwiftPM caches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SIMULATOR_TARGETS = (
    "SimCore", "PhysicsAVBD", "GPUSim", "GPUSimDemos", "GPUSimRenderer"
)
DEVELOPMENT_LAYER_TARGETS = ("Robotics", "RL", "MLXRL")
ENTRY_POINT_TARGETS = ("avbd", "AVBDApp")
DEVELOPMENT_TARGETS = (*DEVELOPMENT_LAYER_TARGETS, *ENTRY_POINT_TARGETS)
ALL_LOCAL_MODULES = frozenset((*SIMULATOR_TARGETS, *DEVELOPMENT_TARGETS))
EXTERNAL_PROJECT_DIRECTORIES = (Path("Robots/Arachne15"),)

MLX_PRODUCTS = ("MLX", "MLXNN", "MLXOptimizers", "MLXRandom", "MLXLinalg")
SUPERSEDED_TARGETS = frozenset(
    {
        "AVBDCore",
        "AVBDLearn",
        "RLCore",
        "RLTasks",
        "PolicyFormat",
        "PolicyRuntime",
        "MLXLearning",
        "RLExperiments",
    }
)


def _target(name: str) -> str:
    return f"target:{name}"


def _product(name: str, package: str) -> str:
    return f"product:{name}@{package}"


SIMULATOR_DEPENDENCIES = {
    "SimCore": frozenset(),
    "PhysicsAVBD": frozenset({_target("SimCore")}),
    "GPUSim": frozenset({_target("SimCore"), _target("PhysicsAVBD")}),
    "GPUSimDemos": frozenset({_target("SimCore")}),
    "GPUSimRenderer": frozenset({_target("SimCore"), _target("PhysicsAVBD")}),
}
SIMULATOR_RESOURCES = {
    "SimCore": frozenset(),
    "PhysicsAVBD": frozenset({"copy:Shaders"}),
    "GPUSim": frozenset(),
    "GPUSimDemos": frozenset({"copy:Assets"}),
    "GPUSimRenderer": frozenset(),
}
SIMULATOR_PRODUCTS = {
    name: ("library", (name,)) for name in SIMULATOR_TARGETS
}

_SIM_CORE = _product("SimCore", "gpu-sim")
_PHYSICS = _product("PhysicsAVBD", "gpu-sim")
_DEMOS = _product("GPUSimDemos", "gpu-sim")
_RENDERER = _product("GPUSimRenderer", "gpu-sim")
DEVELOPMENT_DEPENDENCIES = {
    "Robotics": frozenset({_SIM_CORE}),
    "RL": frozenset({_SIM_CORE, _PHYSICS, _DEMOS, _target("Robotics")}),
    "MLXRL": frozenset(
        {
            _SIM_CORE,
            _PHYSICS,
            _target("Robotics"),
            _target("RL"),
            *(_product(name, "mlx-swift") for name in MLX_PRODUCTS),
        }
    ),
    "avbd": frozenset(
        {
            _SIM_CORE,
            _PHYSICS,
            _DEMOS,
            _target("Robotics"),
            _target("RL"),
            _target("MLXRL"),
        }
    ),
    "AVBDApp": frozenset(
        {
            _SIM_CORE,
            _PHYSICS,
            _DEMOS,
            _RENDERER,
            _target("Robotics"),
            _target("RL"),
            _target("MLXRL"),
        }
    ),
}
DEVELOPMENT_RESOURCES = {
    **{name: frozenset() for name in DEVELOPMENT_TARGETS},
    "Robotics": frozenset({"copy:Assets"}),
}
DEVELOPMENT_PRODUCTS = {
    **{
        name: ("library", (name,)) for name in DEVELOPMENT_LAYER_TARGETS
    },
    **{name: ("executable", (name,)) for name in ENTRY_POINT_TARGETS},
}

SIMULATOR_IMPORTS = {
    "SimCore": frozenset(),
    "PhysicsAVBD": frozenset({"SimCore"}),
    "GPUSim": frozenset({"SimCore", "PhysicsAVBD"}),
    "GPUSimDemos": frozenset({"SimCore"}),
    "GPUSimRenderer": frozenset({"SimCore", "PhysicsAVBD"}),
}
DEVELOPMENT_IMPORTS = {
    "Robotics": frozenset({"SimCore"}),
    "RL": frozenset({"SimCore", "PhysicsAVBD", "GPUSimDemos", "Robotics"}),
    "MLXRL": frozenset({"SimCore", "PhysicsAVBD", "Robotics", "RL"}),
    "avbd": frozenset(
        {"SimCore", "PhysicsAVBD", "GPUSimDemos", "Robotics", "RL", "MLXRL"}
    ),
    "AVBDApp": frozenset(
        {
            "SimCore", "PhysicsAVBD", "GPUSimDemos", "GPUSimRenderer",
            "Robotics", "RL", "MLXRL"
        }
    ),
}


@dataclass(frozen=True)
class ManifestContract:
    package_name: str
    dependencies: Mapping[str, frozenset[str]]
    resources: Mapping[str, frozenset[str]]
    products: Mapping[str, tuple[str, tuple[str, ...]]]
    executable_targets: frozenset[str] = frozenset()


SIMULATOR_CONTRACT = ManifestContract(
    "gpu-sim",
    SIMULATOR_DEPENDENCIES,
    SIMULATOR_RESOURCES,
    SIMULATOR_PRODUCTS,
)
DEVELOPMENT_CONTRACT = ManifestContract(
    "avbd-metal",
    DEVELOPMENT_DEPENDENCIES,
    DEVELOPMENT_RESOURCES,
    DEVELOPMENT_PRODUCTS,
    frozenset(ENTRY_POINT_TARGETS),
)

_IMPORT_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<attributes>(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^\n)]*\))?[ \t]+)*)"
    r"(?:(?P<access>private|fileprivate|internal|package|public)[ \t]+)?"
    r"import[ \t]+"
    r"(?:(?:typealias|struct|class|enum|protocol|let|var|func|operator)[ \t]+)?"
    r"(?P<module>[A-Za-z_][A-Za-z0-9_]*)\b"
)


class VerificationError(RuntimeError):
    """An architecture invariant was violated."""


def _describe(values: Iterable[str]) -> str:
    return "[" + ", ".join(sorted(values)) + "]"


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{description} must be an object")
    return value


def _first_string(value: Any, description: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if value and isinstance(value[0], str):
            return value[0]
    raise VerificationError(f"malformed {description} in SwiftPM package dump")


def _normalize_dependency(raw: Any, target_name: str) -> str:
    dependency = _require_mapping(raw, f"dependency of {target_name}")
    if "byName" in dependency:
        return _target(_first_string(dependency["byName"], "by-name dependency"))
    if "target" in dependency:
        return _target(_first_string(dependency["target"], "target dependency"))
    if "product" in dependency:
        payload = dependency["product"]
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise VerificationError(f"malformed product dependency in {target_name}")
        if len(payload) < 2 or not isinstance(payload[0], str):
            raise VerificationError(f"malformed product dependency in {target_name}")
        package = payload[1]
        if not isinstance(package, str) or not package:
            raise VerificationError(
                f"product dependency {payload[0]} in {target_name} must name its package"
            )
        return _product(payload[0], package)
    raise VerificationError(f"unknown dependency form in {target_name}: {raw!r}")


def _normalize_dependencies(target: Mapping[str, Any]) -> frozenset[str]:
    name = str(target.get("name", "<unnamed>"))
    raw = target.get("dependencies", [])
    if not isinstance(raw, list):
        raise VerificationError(f"dependencies for target {name} must be a list")
    dependencies = [_normalize_dependency(item, name) for item in raw]
    if len(dependencies) != len(set(dependencies)):
        raise VerificationError(f"target {name} declares a dependency more than once")
    return frozenset(dependencies)


def _normalize_resources(target: Mapping[str, Any]) -> frozenset[str]:
    name = str(target.get("name", "<unnamed>"))
    raw_resources = target.get("resources", [])
    if not isinstance(raw_resources, list):
        raise VerificationError(f"resources for target {name} must be a list")
    resources: list[str] = []
    for raw in raw_resources:
        resource = _require_mapping(raw, f"resource of {name}")
        path, rule = resource.get("path"), resource.get("rule")
        if not isinstance(path, str) or not isinstance(rule, Mapping):
            raise VerificationError(f"malformed resource in target {name}: {raw!r}")
        rule_names = list(rule)
        if len(rule_names) != 1 or rule_names[0] not in {"copy", "process"}:
            raise VerificationError(f"unknown resource rule in target {name}: {raw!r}")
        resources.append(f"{rule_names[0]}:{path}")
    if len(resources) != len(set(resources)):
        raise VerificationError(f"target {name} declares a resource more than once")
    return frozenset(resources)


def _product_kind(raw_type: Any, product_name: str) -> str:
    product_type = _require_mapping(raw_type, f"type of product {product_name}")
    if len(product_type) != 1:
        raise VerificationError(f"malformed type for product {product_name}")
    kind = next(iter(product_type))
    if kind not in {"library", "executable"}:
        raise VerificationError(f"unexpected type {kind!r} for product {product_name}")
    return kind


def verify_manifest(
    manifest: Mapping[str, Any], contract: ManifestContract = SIMULATOR_CONTRACT
) -> None:
    """Verify exact production targets, products, dependencies, and resources."""

    if manifest.get("name", contract.package_name) != contract.package_name:
        raise VerificationError(
            f"package must be named {contract.package_name}, got {manifest.get('name')!r}"
        )
    raw_targets, raw_products = manifest.get("targets"), manifest.get("products")
    if not isinstance(raw_targets, list) or not isinstance(raw_products, list):
        raise VerificationError("SwiftPM package dump lacks targets or products")
    targets = {
        str(item.get("name")): _require_mapping(item, "target")
        for item in raw_targets
        if isinstance(item, Mapping)
    }
    products = {
        str(item.get("name")): _require_mapping(item, "product")
        for item in raw_products
        if isinstance(item, Mapping)
    }
    if len(targets) != len(raw_targets) or len(products) != len(raw_products):
        raise VerificationError("SwiftPM package dump has duplicate or malformed names")

    superseded = SUPERSEDED_TARGETS.intersection((*targets, *products))
    if superseded:
        raise VerificationError(
            f"superseded target/product names remain: {_describe(superseded)}"
        )

    production = {
        name for name, target in targets.items()
        if target.get("type") in {"regular", "executable"}
    }
    expected = set(contract.dependencies)
    if production != expected:
        raise VerificationError(
            "production target set differs from package contract; "
            f"missing={_describe(expected - production)}, "
            f"unexpected={_describe(production - expected)}"
        )

    for name, expected_dependencies in contract.dependencies.items():
        target = targets[name]
        expected_type = (
            "executable" if name in contract.executable_targets else "regular"
        )
        if target.get("type") != expected_type:
            raise VerificationError(f"target {name} must be {expected_type}")
        actual_dependencies = _normalize_dependencies(target)
        if actual_dependencies != expected_dependencies:
            raise VerificationError(
                f"target {name} dependencies must be {_describe(expected_dependencies)}, "
                f"got {_describe(actual_dependencies)}"
            )
        actual_resources = _normalize_resources(target)
        if actual_resources != contract.resources[name]:
            raise VerificationError(
                f"target {name} resources must be {_describe(contract.resources[name])}, "
                f"got {_describe(actual_resources)}"
            )
        expected_path = f"Sources/{name}"
        if target.get("path") not in (None, expected_path):
            raise VerificationError(f"target {name} must use {expected_path}")
        if target.get("sources") not in (None, []):
            raise VerificationError(
                f"target {name} must not hide files behind an explicit sources list"
            )
        if target.get("exclude", []):
            raise VerificationError(f"target {name} must not use target excludes")

    if set(products) != set(contract.products):
        raise VerificationError(
            "product set differs from package contract; "
            f"missing={_describe(set(contract.products) - set(products))}, "
            f"unexpected={_describe(set(products) - set(contract.products))}"
        )
    for name, (expected_kind, expected_targets) in contract.products.items():
        product = products[name]
        actual_targets = product.get("targets")
        if (_product_kind(product.get("type"), name) != expected_kind
                or tuple(actual_targets or []) != expected_targets):
            raise VerificationError(
                f"product {name} must be {expected_kind} over {list(expected_targets)}"
            )


def verify_package_dependencies(
    simulator: Mapping[str, Any], development: Mapping[str, Any], root: Path
) -> None:
    """Keep the distributable root hermetic and workspace dependencies exact."""

    if simulator.get("dependencies") != []:
        raise VerificationError("the root GPUSim package must have zero dependencies")
    raw = development.get("dependencies")
    if not isinstance(raw, list) or len(raw) != 2:
        raise VerificationError(
            "Development must depend only on local GPUSim and mlx-swift"
        )
    local_entries, remote_urls = [], []
    for dependency in raw:
        item = _require_mapping(dependency, "package dependency")
        if "fileSystem" in item:
            local_entries.extend(item["fileSystem"])
        elif "sourceControl" in item:
            for entry in item["sourceControl"]:
                location = entry.get("location", {}).get("remote", [])
                if location:
                    remote_urls.append(location[0].get("urlString"))
        else:
            raise VerificationError(
                "Development declares an unknown package dependency"
            )
    if len(local_entries) != 1:
        raise VerificationError(
            "Development must have exactly one local package dependency"
        )
    local = local_entries[0]
    if (local.get("nameForTargetDependencyResolutionOnly") != "gpu-sim"
            or Path(str(local.get("path"))).resolve() != root.resolve()):
        raise VerificationError("Development local dependency must resolve to root GPUSim")
    if remote_urls != ["https://github.com/ml-explore/mlx-swift"]:
        raise VerificationError("Development may resolve only the approved mlx-swift URL")


def _strip_swift_comments(source: str) -> str:
    """Blank Swift comments while preserving offsets and line numbers."""

    result: list[str] = []
    index = block_depth = 0
    in_string = escaped = False
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if block_depth:
            if char == "/" and following == "*":
                result.extend((" ", " "))
                block_depth += 1
                index += 2
            elif char == "*" and following == "/":
                result.extend((" ", " "))
                block_depth -= 1
                index += 2
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if in_string:
            result.append(char)
            if char == "\n":
                in_string = escaped = False
            elif escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == "/" and following == "/":
            result.extend((" ", " "))
            index += 2
            while index < len(source) and source[index] != "\n":
                result.append(" ")
                index += 1
            continue
        if char == "/" and following == "*":
            result.extend((" ", " "))
            block_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
        result.append(char)
        index += 1
    return "".join(result)


def _source_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.swift") if path.is_file())


def _verify_imports_in_target(
    package_root: Path, target_name: str, allowed: frozenset[str]
) -> None:
    target_directory = package_root / "Sources" / target_name
    if not target_directory.is_dir():
        raise VerificationError(f"missing source directory {target_directory}")
    reexports: set[str] = set()
    for source_path in _source_files(target_directory):
        relative = source_path.relative_to(package_root)
        source = _strip_swift_comments(source_path.read_text(encoding="utf-8"))
        for match in _IMPORT_RE.finditer(source):
            module = match.group("module")
            attributes = match.group("attributes")
            access = match.group("access")
            line = source.count("\n", 0, match.start()) + 1
            if "@testable" in attributes:
                raise VerificationError(
                    f"{relative}:{line}: production source cannot use @testable import"
                )
            if "@_exported" in attributes:
                if target_name != "GPUSim":
                    raise VerificationError(
                        f"{relative}:{line}: only GPUSim may re-export modules"
                    )
                reexports.add(module)
            if access == "public":
                raise VerificationError(
                    f"{relative}:{line}: public import does not provide the "
                    "GPUSim single-import contract"
                )
            if module in SUPERSEDED_TARGETS:
                raise VerificationError(
                    f"{relative}:{line}: imports superseded module {module}"
                )
            if module in ALL_LOCAL_MODULES and module not in allowed:
                raise VerificationError(
                    f"{relative}:{line}: target {target_name} may not import {module}; "
                    f"allowed local imports are {_describe(allowed)}"
                )
            if module in MLX_PRODUCTS and target_name != "MLXRL":
                raise VerificationError(
                    f"{relative}:{line}: MLX import {module} belongs in MLXRL"
                )
    if target_name == "GPUSim" and reexports != {"SimCore", "PhysicsAVBD"}:
        raise VerificationError(
            "GPUSim must re-export exactly SimCore and PhysicsAVBD"
        )


def verify_source_imports(root: Path | str) -> None:
    """Verify source ownership and one-way imports across both packages."""

    root = Path(root).resolve()
    development = root / "Development"
    for package_root, contracts in (
        (root, SIMULATOR_IMPORTS), (development, DEVELOPMENT_IMPORTS)
    ):
        sources = package_root / "Sources"
        if not sources.is_dir():
            raise VerificationError(f"missing Sources directory under {package_root}")
        actual_directories = {
            path.name for path in sources.iterdir() if path.is_dir()
        }
        if actual_directories != set(contracts):
            raise VerificationError(
                f"source target directories under {package_root} differ; "
                f"expected={_describe(contracts)}, got={_describe(actual_directories)}"
            )
        for target, allowed in contracts.items():
            _verify_imports_in_target(package_root, target, allowed)

    demo_root = root / "Sources/GPUSimDemos"
    if not (demo_root / "Demos.swift").is_file():
        raise VerificationError("demos must be rooted at Sources/GPUSimDemos/Demos.swift")
    declaration = re.compile(
        r"\b(?:public[ \t]+)?enum[ \t]+Demos\b|\bextension[ \t]+Demos\b"
    )
    for package_root in (root, development):
        for source_path in (package_root / "Sources").rglob("*.swift"):
            if source_path.is_relative_to(demo_root):
                continue
            if declaration.search(_strip_swift_comments(source_path.read_text())):
                raise VerificationError(
                    f"{source_path.relative_to(root)}: demos belong in Sources/GPUSimDemos"
                )


def _is_hidden(path: Path, sources: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(sources).parts)


def verify_resource_layout(root: Path | str) -> None:
    """Require every shader and asset to live under its owning target."""

    root = Path(root).resolve()
    simulator_sources = root / "Sources"
    development_sources = root / "Development/Sources"
    shader_root = simulator_sources / "PhysicsAVBD/Shaders"
    demo_assets = simulator_sources / "GPUSimDemos/Assets"
    robotics_assets = development_sources / "Robotics/Assets"
    for resource_root in (shader_root, demo_assets, robotics_assets):
        if not resource_root.is_dir() or not any(
            path.is_file() for path in resource_root.rglob("*")
        ):
            raise VerificationError(f"missing or empty resource directory {resource_root}")
    for sources in (simulator_sources, development_sources):
        for path in sources.rglob("*"):
            if not path.is_file() or _is_hidden(path, sources):
                continue
            relative = path.relative_to(sources)
            if (path.suffix.lower() == ".metal" or "Shaders" in relative.parts):
                if not path.is_relative_to(shader_root):
                    raise VerificationError(
                        f"{path.relative_to(root)}: shaders belong to PhysicsAVBD/Shaders"
                    )
            if "Assets" in relative.parts and not (
                path.is_relative_to(demo_assets)
                or path.is_relative_to(robotics_assets)
            ):
                raise VerificationError(
                    f"{path.relative_to(root)}: assets belong to GPUSimDemos or Robotics"
                )


def verify_external_project_layout(root: Path | str) -> None:
    """Reject external hardware workspaces that must not be vendored here."""

    root = Path(root).resolve()
    for relative in EXTERNAL_PROJECT_DIRECTORIES:
        if os.path.lexists(root / relative):
            raise VerificationError(
                f"external project directory must not be vendored: {relative}"
            )


def load_package_dump(root: Path | str) -> Mapping[str, Any]:
    """Evaluate one manifest with isolated SwiftPM and compiler caches."""

    root = Path(root).resolve()
    if not (root / "Package.swift").is_file():
        raise VerificationError(f"missing Package.swift under {root}")
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-sim-architecture-") as temporary:
            temporary_path = Path(temporary)
            environment = os.environ.copy()
            environment["SWIFTPM_MODULECACHE_OVERRIDE"] = str(
                temporary_path / "swift-module-cache"
            )
            environment["CLANG_MODULE_CACHE_PATH"] = str(
                temporary_path / "clang-module-cache"
            )
            result = subprocess.run(
                [
                    "swift",
                    "package",
                    "--package-path",
                    str(root),
                    "--disable-sandbox",
                    "--scratch-path",
                    str(temporary_path / "build"),
                    "dump-package",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
    except FileNotFoundError as error:
        raise VerificationError("swift executable was not found") from error
    except subprocess.TimeoutExpired as error:
        raise VerificationError("swift package dump-package timed out") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise VerificationError(f"swift package dump-package failed: {detail}")
    try:
        return _require_mapping(json.loads(result.stdout), "SwiftPM package dump")
    except json.JSONDecodeError as error:
        raise VerificationError(f"SwiftPM returned invalid JSON: {error}") from error


def verify(root: Path | str) -> None:
    root = Path(root).resolve()
    development_root = root / "Development"
    simulator = load_package_dump(root)
    development = load_package_dump(development_root)
    verify_manifest(simulator, SIMULATOR_CONTRACT)
    verify_manifest(development, DEVELOPMENT_CONTRACT)
    verify_package_dependencies(simulator, development, root)
    verify_source_imports(root)
    verify_resource_layout(root)
    verify_external_project_layout(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of Tools)",
    )
    arguments = parser.parse_args(argv)
    try:
        verify(arguments.root)
    except VerificationError as error:
        print(f"architecture verification failed: {error}", file=sys.stderr)
        return 1
    print(
        "verified standalone GPUSim -> optional demos/renderer and "
        "Development/Robotics -> RL -> MLXRL boundaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
