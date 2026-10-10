"""Read-only evaluation API (design decisions D16, D17).

Runs are started from ``ragfabric eval run`` or ``make eval``; a batch takes
minutes to hours on a local model, too long for one HTTP request. These
routes serve the stored runs and the dashboard numbers to the console. Admin
only: evaluation answers and production latency and cost are operator data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_core.evaluation.dashboard import dashboard
from ragfabric_core.evaluation.store import get_run, list_runs, result_dict, run_dict
from ragfabric_server.deps import require_role

router = APIRouter(dependencies=[Depends(require_role("admin"))])


@router.get("/runs")
def runs(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)) -> list[dict]:
    """Recent evaluation runs, newest first, one per target per batch."""
    return [run_dict(r) for r in list_runs(db, limit)]


@router.get("/runs/{run_id}")
def run(run_id: int, db: Session = Depends(get_db)) -> dict:
    """One run with its per question results."""
    found = get_run(db, run_id)
    if found is None:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    run_row, rows = found
    return {**run_dict(run_row), "results": [result_dict(r) for r in rows]}


@router.get("/dashboard")
def dashboard_data(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)) -> dict:
    """Latency percentiles, cost per day, calls per strategy, fallback rate, quality trend."""
    return dashboard(db, days=days)
