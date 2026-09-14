"""Read one retrieval run with its sources and trace (the Trace page's data source)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_server.deps import get_principal
from ragfabric_server.schemas.runs import RunOut, SourceOut

router = APIRouter()


@router.get("/{run_id}", response_model=RunOut)
def get_run(
    run_id: int, db: Session = Depends(get_db), principal: Principal = Depends(get_principal)
) -> RunOut:
    run = db.get(RetrievalRun, run_id)
    if run is None or (principal.role != "admin" and run.user_id != principal.user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    sources = db.query(Source).filter(Source.retrieval_run_id == run.id).order_by(Source.rank).all()
    out = RunOut.model_validate(run)
    out.sources = [SourceOut.model_validate(s) for s in sources]
    return out
