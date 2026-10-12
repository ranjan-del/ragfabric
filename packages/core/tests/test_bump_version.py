"""scripts/bump_version.py: one command moves every version and pin (Phase 10, Task 9)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "bump_version.py"
FILES = [
    "packages/core/pyproject.toml",
    "packages/server/pyproject.toml",
    "packages/cli/pyproject.toml",
    "packages/sdk-python/pyproject.toml",
    "packages/core/src/ragfabric_core/__init__.py",
]
OPTIONAL = ["deploy/compose/docker-compose.prod.yml", "deploy/compose/.env.prod.example"]


@pytest.fixture()
def tree(tmp_path):
    for rel in FILES + OPTIONAL:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def _run(tree, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tree), *args],
        capture_output=True,
        text=True,
    )


def _current():
    import re

    text = (ROOT / FILES[0]).read_text()
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def test_every_version_and_pin_moves(tree):
    old = _current()
    result = _run(tree, "9.8.7")
    assert result.returncode == 0, result.stderr
    for rel in FILES:
        text = (tree / rel).read_text()
        assert old not in text, rel
        assert "9.8.7" in text, rel
    cli = (tree / "packages/cli/pyproject.toml").read_text()
    assert 'ragfabric-core[drive]==9.8.7' in cli and 'ragfabric-sdk==9.8.7' in cli
    assert '__version__ = "9.8.7"' in (tree / FILES[4]).read_text()
    assert "RAGFABRIC_VERSION:-9.8.7" in (tree / OPTIONAL[0]).read_text()
    assert "RAGFABRIC_VERSION=9.8.7" in (tree / OPTIONAL[1]).read_text()


def test_only_versions_change(tree):
    before = {rel: (tree / rel).read_text() for rel in FILES}
    _run(tree, "9.8.7")
    for rel, text in before.items():
        after = (tree / rel).read_text()
        assert after.replace("9.8.7", _current()) == text, rel


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3.4", "1.2.x", "", "1.2.3-beta"])
def test_a_malformed_version_is_refused_and_nothing_changes(tree, bad):
    before = {rel: (tree / rel).read_text() for rel in FILES}
    result = _run(tree, bad)
    assert result.returncode == 2
    assert "not a version" in result.stderr
    assert before == {rel: (tree / rel).read_text() for rel in FILES}


def test_a_prerelease_is_accepted(tree):
    assert _run(tree, "1.0.0rc1").returncode == 0
    assert _run(tree, "--check", "v1.0.0rc1").returncode == 0


def test_running_twice_is_a_no_op(tree):
    _run(tree, "9.8.7")
    snapshot = {rel: (tree / rel).read_text() for rel in FILES + OPTIONAL}
    result = _run(tree, "9.8.7")
    assert result.returncode == 0
    assert "already" in result.stdout
    assert snapshot == {rel: (tree / rel).read_text() for rel in FILES + OPTIONAL}


def test_check_passes_on_a_matching_tag(tree):
    result = _run(tree, "--check", f"v{_current()}")
    assert result.returncode == 0, result.stderr


def test_check_fails_on_a_mismatched_tag(tree):
    result = _run(tree, "--check", "v9.9.9")
    assert result.returncode == 1
    assert "packages/core/pyproject.toml" in result.stderr


def test_check_finds_a_single_missed_pin(tree):
    cli = tree / "packages/cli/pyproject.toml"
    cli.write_text(cli.read_text().replace(f"ragfabric-sdk=={_current()}", "ragfabric-sdk==0.0.1"))
    result = _run(tree, "--check", _current())
    assert result.returncode == 1
    assert "ragfabric-sdk==0.0.1" in result.stderr


def test_the_release_workflow_is_guarded():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    triggers = workflow[True] if True in workflow else workflow["on"]  # YAML 1.1 reads "on" as True
    assert set(triggers) == {"push"} and triggers["push"] == {"tags": ["v*"]}
    jobs = workflow["jobs"]
    verify = " ".join(step.get("run", "") for step in jobs["verify"]["steps"])
    assert "bump_version.py --check" in verify and "GITHUB_REF_NAME" in verify
    assert jobs["images"]["needs"] == "verify" and jobs["dists"]["needs"] == "verify"
    dists = " ".join(step.get("run", "") for step in jobs["dists"]["steps"])
    assert "uv build --all-packages" in dists and "twine check" in dists
    pypi = jobs["pypi"]
    assert pypi["needs"] == "dists"
    assert pypi["if"] == "vars.PYPI_PUBLISH == 'true'"
    assert pypi["environment"]["name"] == "pypi"
    assert pypi["permissions"] == {"id-token": "write"}
    [publish] = [s for s in pypi["steps"] if "gh-action-pypi-publish" in s.get("uses", "")]
    assert publish["uses"].split("@")[1].startswith("v1.")  # a release, never a branch
