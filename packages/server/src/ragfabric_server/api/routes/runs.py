"""Read one retrieval run with its sources and trace (the Trace page's data source).

``GET /ingestion/{id}`` is the ingestion equivalent (Task 16 C1): one
``IngestionRun`` row per ingestion phase, gated the same way a document read
is (``AccessFilter.allows`` against the run's own document), since an
ingestion run is not owned by a principal the way a retrieval run is.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.models.document import Document, IngestionRun
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_server.deps import get_access_filter, get_principal
from ragfabric_server.schemas.runs import IngestionRunOut, RunOut, SourceOut

router = APIRouter()


@router.get("/ingestion/{run_id}", response_model=IngestionRunOut)
def get_ingestion_run(
    run_id: int,
    db: Session = Depends(get_db),
    access: AccessFilter = Depends(get_access_filter),
) -> IngestionRunOut:
    run = db.get(IngestionRun, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    document = db.get(Document, run.document_id) if run.document_id is not None else None
    if document is None or not access.allows(document.id, document.collection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    return IngestionRunOut.model_validate(run)


@router.get("/{run_id}", response_model=RunOut)
def get_run(
    run_id: int, db: Session = Depends(get_db), principal: Principal = Depends(get_principal)
) -> RunOut:
    run = db.get(RetrievalRun, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    if principal.role != "admin":
        if principal.user_id is None or run.user_id != principal.user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    sources = db.query(Source).filter(Source.retrieval_run_id == run.id).order_by(Source.rank).all()
    out = RunOut.model_validate(run)
    out.sources = [SourceOut.model_validate(s) for s in sources]
    return out
