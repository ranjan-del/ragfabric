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
