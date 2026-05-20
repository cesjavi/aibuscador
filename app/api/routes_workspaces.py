from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Workspace
from app.services.rag_service import delete_workspace


router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""


@router.post("")
def create_workspace(payload: WorkspaceCreateRequest, db: Session = Depends(get_db)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="El nombre del workspace no puede estar vacío.")
    existing = db.query(Workspace).filter(Workspace.name == name).first()
    if existing:
        raise HTTPException(status_code=409, detail="Ya existe un workspace con ese nombre.")
    workspace = Workspace(name=name, description=payload.description.strip())
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    return _serialize_workspace(workspace)


@router.get("")
def list_workspaces(db: Session = Depends(get_db)) -> list[dict]:
    workspaces = db.query(Workspace).order_by(Workspace.created_at.desc()).all()
    return [_serialize_workspace(workspace) for workspace in workspaces]


@router.delete("/{workspace_id}")
def remove_workspace(workspace_id: int, db: Session = Depends(get_db)) -> dict:
    if not delete_workspace(db, workspace_id):
        raise HTTPException(status_code=404, detail="Workspace no encontrado.")
    return {"deleted": True, "workspace_id": workspace_id}


def _serialize_workspace(workspace: Workspace) -> dict:
    return {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "created_at": workspace.created_at.isoformat(),
        "documents": len(workspace.documents),
    }
