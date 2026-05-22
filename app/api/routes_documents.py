from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Document
from app.services.document_loader import extract_text_from_path, iter_supported_files, save_upload
from app.services.rag_service import (
    DuplicateDocumentError,
    delete_document,
    delete_duplicate_documents,
    ingest_document,
)


router = APIRouter(prefix="/documents", tags=["documents"])


class TextIngestRequest(BaseModel):
    workspace_id: int
    name: str
    text: str
    source: str = "texto_manual"


class FolderIngestRequest(BaseModel):
    workspace_id: int
    folder_path: str


@router.post("/upload")
async def upload_document(
    workspace_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    path: Path | None = None
    try:
        from app.config import get_settings

        path = await save_upload(file, get_settings().upload_dir)
        content = extract_text_from_path(path)
        document = ingest_document(
            db=db,
            workspace_id=workspace_id,
            name=file.filename or path.name,
            file_type=Path(path).suffix.lower().lstrip("."),
            source=str(path),
            content=content,
        )
        return {"id": document.id, "name": document.name, "chunks": len(document.chunks)}
    except Exception as exc:
        if path:
            path.unlink(missing_ok=True)
        if isinstance(exc, DuplicateDocumentError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"Error al procesar documento: {exc}") from exc


@router.post("/text")
def ingest_text(payload: TextIngestRequest, db: Session = Depends(get_db)) -> dict:
    try:
        document = ingest_document(db, payload.workspace_id, payload.name, "text", payload.source, payload.text)
        return {"id": document.id, "name": document.name, "chunks": len(document.chunks)}
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_documents(workspace_id: int | None = None, db: Session = Depends(get_db)) -> list[dict]:
    query = db.query(Document)
    if workspace_id is not None:
        query = query.filter(Document.workspace_id == workspace_id)
    documents = query.order_by(Document.created_at.desc()).all()
    return [
        {
            "id": document.id,
            "workspace_id": document.workspace_id,
            "workspace_name": document.workspace.name,
            "name": document.name,
            "file_type": document.file_type,
            "source": document.source,
            "created_at": document.created_at.isoformat(),
            "chunks": len(document.chunks),
        }
        for document in documents
    ]


@router.post("/folder")
def ingest_folder(payload: FolderIngestRequest, db: Session = Depends(get_db)) -> dict:
    try:
        files = iter_supported_files(Path(payload.folder_path))
        loaded: list[dict] = []
        errors: list[dict] = []
        for path in files:
            try:
                content = extract_text_from_path(path)
                document = ingest_document(
                    db=db,
                    workspace_id=payload.workspace_id,
                    name=path.name,
                    file_type=path.suffix.lower().lstrip("."),
                    source=str(path),
                    content=content,
                )
                loaded.append({"id": document.id, "name": document.name, "chunks": len(document.chunks)})
            except DuplicateDocumentError as exc:
                errors.append({"path": str(path), "error": str(exc), "skipped": True})
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return {"loaded": loaded, "errors": errors, "total_found": len(files)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo recorrer la carpeta: {exc}") from exc


@router.delete("/duplicates")
def remove_duplicate_documents(workspace_id: int | None = None, db: Session = Depends(get_db)) -> dict:
    return delete_duplicate_documents(db, workspace_id)


@router.delete("/{document_id}")
def remove_document(document_id: int, db: Session = Depends(get_db)) -> dict:
    if not delete_document(db, document_id):
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    return {"deleted": True, "document_id": document_id}
