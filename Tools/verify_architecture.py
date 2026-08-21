#!/usr/bin/env python3
"""Verify the package and source boundaries of the five-layer architecture.

The verifier intentionally uses only the Python standard library.  SwiftPM's
``dump-package`` command evaluates Package.swift without resolving or fetching
package dependencies, and all of its caches are placed in a temporary folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


LAYER_TARGETS = ("SimCore", "PhysicsAVBD", "Robotics", "RL", "MLXRL")
ENTRY_POINT_TARGETS = ("avbd", "AVBDApp")
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


EXPECTED_TARGET_DEPENDENCIES = {
    "SimCore": frozenset(),
    "PhysicsAVBD": frozenset({_target("SimCore")}),
    "Robotics": frozenset({_target("SimCore")}),
    # PhysicsAVBD is an intentional edge while concrete physical-flow and
    # simulator-backed environment APIs live in RL.
    "RL": frozenset(
        {_target("SimCore"), _target("PhysicsAVBD"), _target("Robotics")}
    ),
    "MLXRL": frozenset(
        {
            _target("SimCore"),
            _target("PhysicsAVBD"),
            _target("Robotics"),
            _target("RL"),
            *(_product(name, "mlx-swift") for name in MLX_PRODUCTS),
        }
    ),
    "avbd": frozenset({_target(name) for name in LAYER_TARGETS}),
    "AVBDApp": frozenset({_target(name) for name in LAYER_TARGETS}),
}

EXPECTED_TARGET_TYPES = {
    **{name: "regular" for name in LAYER_TARGETS},
    **{name: "executable" for name in ENTRY_POINT_TARGETS},
}

EXPECTED_RESOURCES = {
    **{name: frozenset() for name in EXPECTED_TARGET_DEPENDENCIES},
    "PhysicsAVBD": frozenset({"copy:Shaders", "copy:Assets"}),
    "Robotics": frozenset({"copy:Assets"}),
}

EXPECTED_PRODUCTS = {
    **{name: ("library", (name,)) for name in LAYER_TARGETS},
    **{name: ("executable", (name,)) for name in ENTRY_POINT_TARGETS},
}

ALLOWED_LOCAL_IMPORTS = {
    "SimCore": frozenset(),
    "PhysicsAVBD": frozenset({"SimCore"}),
    "Robotics": frozenset({"SimCore"}),
    "RL": frozenset({"SimCore", "PhysicsAVBD", "Robotics"}),
    "MLXRL": frozenset({"SimCore", "PhysicsAVBD", "Robotics", "RL"}),
    "avbd": frozenset(LAYER_TARGETS),
    "AVBDApp": frozenset(LAYER_TARGETS),
}

ALL_LOCAL_MODULES = frozenset(
    {*LAYER_TARGETS, *ENTRY_POINT_TARGETS, *SUPERSEDED_TARGETS}
)

_IMPORT_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<attributes>(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^\n)]*\))?[ \t]+)*)"
    r"import[ \t]+"
    r"(?:(?:typealias|struct|class|enum|protocol|let|var|func|operator)[ \t]+)?"
    r"(?P<module>[A-Za-z_][A-Za-z0-9_]*)\b"
)
class VerificationError(RuntimeError):
    """An architecture invariant was violated."""


def _describe(values: Iterable[str]) -> str:
    values = sorted(values)
    return "[" + ", ".join(values) + "]"


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
            raise VerificationError(
                f"malformed product dependency in target {target_name}"
            )
        if len(payload) < 2 or not isinstance(payload[0], str):
            raise VerificationError(
                f"malformed product dependency in target {target_name}"
            )
        package = payload[1]
        if not isinstance(package, str) or not package:
            raise VerificationError(
                f"product dependency {payload[0]} in target {target_name} "
                "must name its package"
            )
        return _product(payload[0], package)
    raise VerificationError(f"unknown dependency form in target {target_name}: {raw!r}")


def _normalize_dependencies(target: Mapping[str, Any]) -> frozenset[str]:
    name = target.get("name", "<unnamed>")
    raw_dependencies = target.get("dependencies", [])
    if not isinstance(raw_dependencies, list):
        raise VerificationError(f"dependencies for target {name} must be a list")
    dependencies = [_normalize_dependency(raw, str(name)) for raw in raw_dependencies]
    if len(dependencies) != len(set(dependencies)):
        raise VerificationError(f"target {name} declares a dependency more than once")
    return frozenset(dependencies)


def _normalize_resources(target: Mapping[str, Any]) -> frozenset[str]:
    name = target.get("name", "<unnamed>")
    raw_resources = target.get("resources", [])
    if not isinstance(raw_resources, list):
        raise VerificationError(f"resources for target {name} must be a list")

    resources: list[str] = []
    for raw in raw_resources:
        resource = _require_mapping(raw, f"resource of {name}")
        path = resource.get("path")
        rule = resource.get("rule")
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


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    """Verify exact target/product dependencies from ``swift package dump-package``."""

    raw_targets = manifest.get("targets")
    raw_products = manifest.get("products")
    if not isinstance(raw_targets, list) or not isinstance(raw_products, list):
        raise VerificationError("SwiftPM package dump lacks targets or products")

    targets: dict[str, Mapping[str, Any]] = {}
    for raw in raw_targets:
        target = _require_mapping(raw, "target")
        name = target.get("name")
        if not isinstance(name, str) or not name:
            raise VerificationError("SwiftPM package dump contains an unnamed target")
        if name in targets:
            raise VerificationError(f"duplicate target {name}")
        targets[name] = target

    products: dict[str, Mapping[str, Any]] = {}
    for raw in raw_products:
        product = _require_mapping(raw, "product")
        name = product.get("name")
        if not isinstance(name, str) or not name:
            raise VerificationError("SwiftPM package dump contains an unnamed product")
        if name in products:
            raise VerificationError(f"duplicate product {name}")
        products[name] = product

    superseded_targets = SUPERSEDED_TARGETS.intersection(targets)
    superseded_products = SUPERSEDED_TARGETS.intersection(products)
    if superseded_targets or superseded_products:
        found = superseded_targets.union(superseded_products)
        raise VerificationError(
            f"superseded target/product names remain: {_describe(found)}"
        )

    production_targets = {
        name
        for name, target in targets.items()
        if target.get("type") in {"regular", "executable"}
    }
    expected_targets = set(EXPECTED_TARGET_DEPENDENCIES)
    if production_targets != expected_targets:
        missing = expected_targets - production_targets
        unexpected = production_targets - expected_targets
        raise VerificationError(
            "production target set differs from the five-layer architecture; "
            f"missing={_describe(missing)}, unexpected={_describe(unexpected)}"
        )

    for name, expected_dependencies in EXPECTED_TARGET_DEPENDENCIES.items():
        target = targets[name]
        actual_type = target.get("type")
        if actual_type != EXPECTED_TARGET_TYPES[name]:
            raise VerificationError(
                f"target {name} must be {EXPECTED_TARGET_TYPES[name]}, got {actual_type!r}"
            )

        actual_dependencies = _normalize_dependencies(target)
        if actual_dependencies != expected_dependencies:
            raise VerificationError(
                f"target {name} dependencies must be "
                f"{_describe(expected_dependencies)}, got "
                f"{_describe(actual_dependencies)}"
            )

        actual_resources = _normalize_resources(target)
        expected_resources = EXPECTED_RESOURCES[name]
        if actual_resources != expected_resources:
            raise VerificationError(
                f"target {name} resources must be {_describe(expected_resources)}, "
                f"got {_describe(actual_resources)}"
            )

        expected_path = f"Sources/{name}"
        explicit_path = target.get("path")
        if explicit_path not in (None, expected_path):
            raise VerificationError(
                f"target {name} must use {expected_path}, got {explicit_path!r}"
            )
        if target.get("sources") not in (None, []):
            raise VerificationError(
                f"target {name} must not hide files behind an explicit sources list"
            )
        if target.get("exclude", []):
            raise VerificationError(
                f"target {name} must not hide files behind target excludes"
            )

    if set(products) != set(EXPECTED_PRODUCTS):
        missing = set(EXPECTED_PRODUCTS) - set(products)
        unexpected = set(products) - set(EXPECTED_PRODUCTS)
        raise VerificationError(
            "product set differs from the five-layer architecture; "
            f"missing={_describe(missing)}, unexpected={_describe(unexpected)}"
        )

    for name, (expected_kind, expected_targets_for_product) in EXPECTED_PRODUCTS.items():
        product = products[name]
        actual_kind = _product_kind(product.get("type"), name)
        raw_product_targets = product.get("targets")
        if not isinstance(raw_product_targets, list) or not all(
            isinstance(target, str) for target in raw_product_targets
        ):
            raise VerificationError(f"product {name} has malformed targets")
        actual_product_targets = tuple(raw_product_targets)
        if (
            actual_kind != expected_kind
            or actual_product_targets != expected_targets_for_product
        ):
            raise VerificationError(
                f"product {name} must be {expected_kind} over "
                f"{list(expected_targets_for_product)}, got {actual_kind} over "
                f"{list(actual_product_targets)}"
            )


def _strip_swift_comments(source: str) -> str:
    """Replace Swift comments with whitespace while preserving line numbers.

    Swift permits nested block comments, so a regular expression is not enough.
    String literals are copied verbatim so comment markers inside strings do not
    accidentally erase surrounding source.
    """

    result: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    escaped = False

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
                # This also makes the common single-line string case recover
                # gracefully when scanning partially edited source.
                in_string = False
                escaped = False
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


def _source_files(target_directory: Path) -> list[Path]:
    return sorted(path for path in target_directory.rglob("*.swift") if path.is_file())


def _line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _verify_imports_in_target(root: Path, target_name: str) -> None:
    target_directory = root / "Sources" / target_name
    if not target_directory.is_dir():
        raise VerificationError(f"missing source directory Sources/{target_name}")

    allowed_local_imports = ALLOWED_LOCAL_IMPORTS[target_name]
    for source_path in _source_files(target_directory):
        relative_path = source_path.relative_to(root)
        source = _strip_swift_comments(source_path.read_text(encoding="utf-8"))
        for match in _IMPORT_RE.finditer(source):
            module = match.group("module")
            attributes = match.group("attributes")
            line = _line_number(source, match.start())

            if "@testable" in attributes:
                raise VerificationError(
                    f"{relative_path}:{line}: production source cannot use @testable import"
                )
            if "@_exported" in attributes:
                raise VerificationError(
                    f"{relative_path}:{line}: production modules may not re-export modules"
                )
            if module in SUPERSEDED_TARGETS:
                raise VerificationError(
                    f"{relative_path}:{line}: imports superseded module {module}"
                )
            if module in ALL_LOCAL_MODULES and module not in allowed_local_imports:
                raise VerificationError(
                    f"{relative_path}:{line}: target {target_name} may not import {module}; "
                    f"allowed local imports are {_describe(allowed_local_imports)}"
                )
            if module in MLX_PRODUCTS and target_name != "MLXRL":
                raise VerificationError(
                    f"{relative_path}:{line}: MLX import {module} belongs in MLXRL"
                )


def verify_source_imports(root: Path | str) -> None:
    """Verify source-level direction and reject superseded modules."""

    root = Path(root).resolve()
    sources = root / "Sources"
    if not sources.is_dir():
        raise VerificationError(f"missing Sources directory under {root}")

    for name in sorted(SUPERSEDED_TARGETS):
        if (sources / name).exists():
            raise VerificationError(f"superseded source directory remains: Sources/{name}")

    for target_name in ALLOWED_LOCAL_IMPORTS:
        _verify_imports_in_target(root, target_name)

    demo_root = sources / "PhysicsAVBD" / "Demos"
    if not (demo_root / "Demos.swift").is_file():
        raise VerificationError(
            "AVBD-tuned demos must be rooted at Sources/PhysicsAVBD/Demos/Demos.swift"
        )
    demo_declaration = re.compile(r"\b(?:public[ \t]+)?enum[ \t]+Demos\b|\bextension[ \t]+Demos\b")
    for source_path in sorted(sources.rglob("*.swift")):
        if _is_beneath(source_path, demo_root):
            continue
        source = _strip_swift_comments(source_path.read_text(encoding="utf-8"))
        if source_path.name.startswith("Demos") or demo_declaration.search(source):
            raise VerificationError(
                f"{source_path.relative_to(root)}: AVBD-tuned demos belong in "
                "Sources/PhysicsAVBD/Demos"
            )


def _is_beneath(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _is_hidden(path: Path, sources: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(sources).parts)


def verify_resource_layout(root: Path | str) -> None:
    """Verify that simulator shaders and robotics assets have single owners."""

    root = Path(root).resolve()
    sources = root / "Sources"
    shader_root = sources / "PhysicsAVBD" / "Shaders"
    physics_asset_root = sources / "PhysicsAVBD" / "Assets"
    robotics_asset_root = sources / "Robotics" / "Assets"

    for resource_root in (shader_root, physics_asset_root, robotics_asset_root):
        if not resource_root.is_dir():
            raise VerificationError(
                f"missing owned resource directory {resource_root.relative_to(root)}"
            )
        if not any(
            path.is_file() and not _is_hidden(path, sources)
            for path in resource_root.rglob("*")
        ):
            raise VerificationError(
                f"owned resource directory {resource_root.relative_to(root)} is empty"
            )

    for path in sources.rglob("*"):
        if not path.is_file() or _is_hidden(path, sources):
            continue
        relative = path.relative_to(sources)
        in_shader_directory = "Shaders" in relative.parts
        in_asset_directory = "Assets" in relative.parts

        if (path.suffix.lower() == ".metal" or in_shader_directory) and not _is_beneath(
            path, shader_root
        ):
            raise VerificationError(
                f"{path.relative_to(root)}: shader resources belong to "
                "Sources/PhysicsAVBD/Shaders"
            )
        if in_asset_directory and not (
            _is_beneath(path, physics_asset_root)
            or _is_beneath(path, robotics_asset_root)
        ):
            raise VerificationError(
                f"{path.relative_to(root)}: runtime assets belong to their owning "
                "target's Assets directory"
            )


def verify_external_project_layout(root: Path | str) -> None:
    """Reject hardware/design workspaces that must remain separate projects."""

    root = Path(root).resolve()
    for relative in EXTERNAL_PROJECT_DIRECTORIES:
        if os.path.lexists(root / relative):
            raise VerificationError(
                f"external project directory must not be vendored: {relative}"
            )


def load_package_dump(root: Path | str) -> Mapping[str, Any]:
    """Return a SwiftPM package dump without using repository build caches."""

    root = Path(root).resolve()
    if not (root / "Package.swift").is_file():
        raise VerificationError(f"missing Package.swift under {root}")

    try:
        with tempfile.TemporaryDirectory(prefix="avbd-architecture-") as temporary:
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
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise VerificationError(
            f"swift package dump-package returned invalid JSON: {error}"
        ) from error
    return _require_mapping(manifest, "SwiftPM package dump")


def verify(root: Path | str) -> None:
    root = Path(root).resolve()
    verify_manifest(load_package_dump(root))
    verify_source_imports(root)
    verify_resource_layout(root)
    verify_external_project_layout(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the SimCore/PhysicsAVBD/Robotics/RL/MLXRL boundaries."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="package root (defaults to the parent of Tools)",
    )
    arguments = parser.parse_args(argv)

    try:
        verify(arguments.root)
    except VerificationError as error:
        print(f"architecture verification failed: {error}", file=sys.stderr)
        return 1

    print("verified architecture boundaries: SimCore -> PhysicsAVBD/Robotics -> RL -> MLXRL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
