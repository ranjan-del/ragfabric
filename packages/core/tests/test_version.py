"""The package version has one source of truth: packages/core/pyproject.toml.

``ragfabric_core.__version__`` is what ``ragfabric --version`` and the API
report, and it drifted from the packaged version once already. Reading the
pyproject here makes the next drift fail the suite instead of shipping.
"""

import tomllib
from pathlib import Path

import ragfabric_core

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_dunder_version_matches_the_packaged_version():
    packaged = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert ragfabric_core.__version__ == packaged


PACKAGES = PYPROJECT.parents[1]
INTERNAL = {"ragfabric", "ragfabric-core", "ragfabric-server", "ragfabric-sdk"}


def _project(name: str) -> dict:
    return tomllib.loads((PACKAGES / name / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def test_every_package_shares_one_version():
    versions = {name: _project(name)["version"] for name in ("core", "server", "cli", "sdk-python")}
    assert len(set(versions.values())) == 1, versions


def test_internal_dependencies_are_pinned_to_that_version():
    # Unpinned, `pip install ragfabric==X` would pull whatever ragfabric-core is newest on
    # PyPI, so an old install breaks the day a newer core ships.
    version = _project("core")["version"]
    for name in ("core", "server", "cli", "sdk-python"):
        project = _project(name)
        requirements = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
        for requirement in requirements:
            base = requirement.split("==")[0].split("[")[0].strip()
            if base in INTERNAL:
                assert requirement.endswith(f"=={version}"), (name, requirement)
