from ragfabric_cli.ui.panels import render_answer

CITATIONS = [{"marker": "[1]", "used": True, "filename": "leave.pdf", "page": 2, "document_id": 4}]


def test_plain_answer_keeps_todays_lines(capsys):
    render_answer(
        "Ten days carry forward [1].",
        CITATIONS,
        strategy="traditional",
        router={"source": "signals", "reasoning": "A short question about one concept."},
        fallback_from=None,
        subgraph=None,
        dropped_claims=[],
        dropped_relationship_claims=[],
    )
    out = capsys.readouterr().out
    assert "Ten days carry forward [1]." in out
    assert "sources:\n  [1] leave.pdf p2" in out
    assert "Strategy: traditional (signals). A short question about one concept." in out
    assert "\x1b[" not in out


def test_rich_answer_on_a_narrow_terminal_does_not_crash(monkeypatch, capsys):
    from ragfabric_cli.ui import console

    monkeypatch.setattr(console, "is_rich", lambda stream=None: True)
    monkeypatch.setenv("COLUMNS", "40")
    console.get_console.cache_clear()
    try:
        render_answer(
            "x " * 200,
            CITATIONS,
            strategy="graph",
            router=None,
            fallback_from=None,
            subgraph={
                "nodes": [{"id": 1, "name": "Ravi Sharma"}, {"id": 2, "name": "Asha Rao"}],
                "edges": [{"source_id": 1, "target_id": 2, "walked_as": "REPORTS_TO"}],
            },
            dropped_claims=[],
            dropped_relationship_claims=[],
        )
    finally:
        console.get_console.cache_clear()
    out = capsys.readouterr().out
    assert "Answer" in out and "Sources" in out and "leave.pdf" in out
    assert max(len(line) for line in out.splitlines()) <= 40
