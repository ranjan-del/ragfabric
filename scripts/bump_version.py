#!/usr/bin/env python3
"""Move every RagFabric version in lockstep, or check that a tag matches them.

    python scripts/bump_version.py 0.5.0           # set every version and pin to 0.5.0
    python scripts/bump_version.py --check v0.5.0  # exit 1 unless everything says 0.5.0

The four packages release together with exact pins on each other, so a release
touches the same places every time:

- ``version`` in the four ``pyproject.toml`` files,
- every ``ragfabric-<package>==X`` pin (dependencies and extras),
- ``__version__`` in ``ragfabric_core/__init__.py`` (the API reports it),
- the default image version in the production compose file and its ``.env``
  example, when they exist.

Only those strings change. Run ``uv lock`` afterwards so ``uv.lock`` records the
new versions. ``--check`` is what the release workflow runs against the tag, so
a tag that does not match the code never builds anything. Standard library
only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VERSION = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")

PYPROJECTS = [
    "packages/core/pyproject.toml",
    "packages/server/pyproject.toml",
    "packages/cli/pyproject.toml",
    "packages/sdk-python/pyproject.toml",
]
INIT = "packages/core/src/ragfabric_core/__init__.py"
OPTIONAL = {
    "deploy/compose/docker-compose.prod.yml": re.compile(r"(RAGFABRIC_VERSION:-)([^}]+)(\})"),
    "deploy/compose/.env.prod.example": re.compile(r"^(RAGFABRIC_VERSION=)(\S*)()$", re.M),
}

PROJECT_VERSION = re.compile(r'^(version = ")([^"]+)(")', re.M)
PIN = re.compile(r"(ragfabric-(?:core|server|sdk)(?:\[[a-z0-9_,-]+\])?==)([^\"',\s]+)()")
DUNDER = re.compile(r'^(__version__ = ")([^"]+)(")', re.M)


def _rules(root: Path) -> list[tuple[str, re.Pattern[str], int]]:
    """(file, pattern, maximum replacements; 0 means all)."""
    rules: list[tuple[str, re.Pattern[str], int]] = []
    for rel in PYPROJECTS:
        rules.append((rel, PROJECT_VERSION, 1))
        rules.append((rel, PIN, 0))
    rules.append((INIT, DUNDER, 1))
    for rel, pattern in OPTIONAL.items():
        if (root / rel).is_file():
            rules.append((rel, pattern, 0))
    return rules


def check(root: Path, version: str) -> list[str]:
    problems = []
    for rel, pattern, count in _rules(root):
        text = (root / rel).read_text(encoding="utf-8")
        matches = list(pattern.finditer(text))
        if count:
            matches = matches[:count]
        if not matches and rel in PYPROJECTS + [INIT] and pattern is not PIN:
            problems.append(f"{rel}: no version found")
        for m in matches:
            if m.group(2) != version:
                problems.append(f"{rel}: {m.group(0).strip()}")
    return problems


def bump(root: Path, version: str) -> list[str]:
    changed = []
    for rel in dict.fromkeys(rel for rel, _, _ in _rules(root)):
        path = root / rel
        text = original = path.read_text(encoding="utf-8")
        for rule_rel, pattern, count in _rules(root):
            if rule_rel == rel:
                text = pattern.sub(lambda m: m.group(1) + version + m.group(3), text, count=count)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(rel)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", nargs="?", help="the new version, e.g. 0.5.0 or 1.0.0rc1")
    parser.add_argument("--check", metavar="TAG", help="verify everything matches TAG (v optional)")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    root = Path(args.root)

    wanted = args.check if args.check is not None else args.version
    if wanted is None:
        parser.error("give a version, or --check TAG")
    if args.check is not None and wanted.startswith("v"):
        wanted = wanted[1:]
    if not VERSION.match(wanted):
        print(
            f"{wanted!r} is not a version: use X.Y.Z, or X.Y.ZaN, X.Y.ZbN, X.Y.ZrcN "
            "(and vX.Y.Z only as a tag with --check)",
            file=sys.stderr,
        )
        return 2

    if args.check is not None:
        problems = check(root, wanted)
        if problems:
            print(
                f"tag {args.check} does not match the code; these say otherwise:", file=sys.stderr
            )
            for line in problems:
                print(f"  {line}", file=sys.stderr)
            print(f"fix with: python scripts/bump_version.py {wanted}", file=sys.stderr)
            return 1
        print(f"every version and pin is {wanted}")
        return 0

    changed = bump(root, wanted)
    if not changed:
        print(f"already {wanted}; nothing changed")
        return 0
    for rel in changed:
        print(f"updated {rel}")
    print("next: uv lock, then commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
