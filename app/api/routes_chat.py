from fastapi import APIRouter, Depends, HTTPException
import httpx
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.services.rag_service import answer_question


router = APIRouter(prefix="/chat", tags=["chat"])


class QueryRequest(BaseModel):
    workspace_id: int
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/query")
async def query(payload: QueryRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return await answer_question(db, payload.workspace_id, payload.question, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"El proveedor LLM devolvió un error: {detail}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar con el proveedor LLM: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al consultar el LLM: {exc}") from exc
