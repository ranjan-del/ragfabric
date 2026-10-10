"""Write the server's OpenAPI specification, deterministically.

``python -m ragfabric_server.openapi [PATH]`` prints the specification, or
writes it to PATH. The TypeScript SDK is generated from the checked in copy
at ``packages/sdk-typescript/openapi.json`` (Phase 9 design, decision D2),
and ``tests/test_openapi_snapshot.py`` fails whenever the live specification
and that copy disagree.

Keys are sorted and the indent is fixed so that the same server always
produces the same bytes: a regenerated snapshot then differs from the old one
only where the API really changed, which is what a reviewer needs to see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def export_spec() -> dict:
    """The application's OpenAPI document as a dict.

    The app is imported here rather than at module scope so that importing
    this module (for example from a test that only needs ``spec_text``) does
    not build the application before the caller has set its environment.
    """
    from ragfabric_server.main import app

    return app.openapi()


def spec_text() -> str:
    """The specification as stable JSON text with a trailing newline."""
    return json.dumps(export_spec(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = spec_text()
    if args:
        Path(args[0]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
