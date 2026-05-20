from typing import Any

import chromadb
from chromadb.config import Settings

from app.config import get_settings


class VectorStore:
    """Small ChromaDB wrapper for local semantic search."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: list[float], top_k: int, workspace_id: int | None = None) -> list[dict[str, Any]]:
        where = {"workspace_id": workspace_id} if workspace_id is not None else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        return [
            {"id": ids[i], "text": documents[i], "metadata": metadatas[i], "distance": distances[i]}
            for i in range(len(documents))
        ]

    def delete_document(self, document_id: int) -> None:
        self.collection.delete(where={"document_id": document_id})

    def delete_workspace(self, workspace_id: int) -> None:
        self.collection.delete(where={"workspace_id": workspace_id})


def get_vector_store() -> VectorStore:
    return VectorStore()
