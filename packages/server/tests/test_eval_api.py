"""The read-only evaluation API."""

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.models.evaluation import EvaluationResult, EvaluationRun


def _seed() -> int:
    with SessionLocal() as db:
        run = EvaluationRun(
            name="b1",
            strategy="traditional",
            question_set="ragfabric-shipped",
            judge_model="lexical",
            summary={"status": "finished", "metrics": {"hit": 1.0}},
        )
        db.add(run)
        db.flush()
        db.add(
            EvaluationResult(
                evaluation_run_id=run.id,
                question_id="q-001",
                question_type="simple_factual",
                question="q",
                hit=True,
                details={"error": None},
            )
        )
        db.commit()
        return run.id


def test_admin_lists_and_reads_runs(client, admin_headers):
    run_id = _seed()
    listed = client.get("/api/eval/runs", headers=admin_headers)
    assert listed.status_code == 200
    assert any(r["id"] == run_id and r["target"] == "traditional" for r in listed.json())
    one = client.get(f"/api/eval/runs/{run_id}", headers=admin_headers).json()
    assert one["batch"] == "b1" and one["results"][0]["question_id"] == "q-001"
    assert client.get("/api/eval/runs/999999", headers=admin_headers).status_code == 404


def test_dashboard_shape(client, admin_headers):
    out = client.get("/api/eval/dashboard?days=7", headers=admin_headers).json()
    assert set(out) == {
        "window_days",
        "since",
        "runs",
        "latency_ms",
        "calls_per_strategy",
        "cost_per_day",
        "fallback_rate",
        "quality_trend",
    }
    assert out["window_days"] == 7


def test_a_non_admin_is_refused(client, auth_headers):
    assert client.get("/api/eval/runs", headers=auth_headers).status_code == 403
    assert client.get("/api/eval/dashboard", headers=auth_headers).status_code == 403
