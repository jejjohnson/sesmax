"""Rules that hold across the workspace's packages.

The tiers of ``docs/design/workspace_and_core.md`` are enforced here: the core
imports no component and nothing from somax, and the umbrella holds no code.
The import check reads the sources rather than importing them, so it also
catches an import inside a function body.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

import sesmax
import sesmax_core


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
if not PACKAGES.is_dir():
    # A built distribution ships tests/ without the workspace.
    pytest.skip("packages/ is not present", allow_module_level=True)

# Top-level imports each package must never make: anything above it in the
# tiers, and somax, which sesmax replaces.
FORBIDDEN: dict[str, set[str]] = {
    "sesmax_core": {"sesmax", "sesmax_ocean", "sesmax_atmos", "somax"},
}


def _imported_roots(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_imports_only_down_the_tiers(package: str) -> None:
    sources = sorted(PACKAGES.glob(f"*/src/{package}/**/*.py"))
    assert sources, f"no sources found for {package}"
    offending = {
        str(path.relative_to(ROOT)): sorted(_imported_roots(path) & FORBIDDEN[package])
        for path in sources
        if _imported_roots(path) & FORBIDDEN[package]
    }
    assert not offending, f"{package} imports above its tier: {offending}"


def test_umbrella_holds_no_code() -> None:
    assert sesmax.__all__ == ["__version__"]


@pytest.mark.parametrize("module", [sesmax, sesmax_core])
def test_version_matches_pyproject(module: object) -> None:
    """``__version__`` and ``pyproject.toml`` move together (release-please)."""
    name = module.__name__  # type: ignore[attr-defined]
    dist = "sesmax" if name == "sesmax" else name.replace("_", "-")
    data = tomllib.loads((PACKAGES / dist / "pyproject.toml").read_text("utf-8"))
    assert module.__version__ == data["project"]["version"]  # type: ignore[attr-defined]
