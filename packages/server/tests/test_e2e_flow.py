"""End to end on SQLite: upload to cited answer through the SDK (Phase 10, Task 7).

Runs in the required "Backend tests" job with the offline embedder and the
deterministic citing model the other server tests use (conftest.py). The
scenario itself lives in ``e2e_scenario.py`` so the PostgreSQL tier runs the
very same steps.
"""

from __future__ import annotations

from e2e_scenario import run_scenario


def test_upload_to_cited_answer_on_sqlite(client, tmp_path):
    outcome = run_scenario(client, tmp_path)
    assert len(outcome.document_ids) == 2
    assert outcome.run_id is not None
    assert outcome.strategy == "traditional"
