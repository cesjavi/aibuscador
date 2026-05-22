from hashlib import sha256
import logging
import re
import unicodedata

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import Chunk, Document, Workspace
from app.services.embedding_service import embed_query, embed_texts
from app.services.llm_service import LLMService
from app.services.text_splitter import clean_text, count_tokens, split_text
from app.services.vector_store import get_vector_store


logger = logging.getLogger("rag.query")


class DuplicateDocumentError(ValueError):
    pass


def content_digest(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def normalize_for_search(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def query_terms(question: str) -> list[str]:
    normalized = normalize_for_search(question)
    terms = re.findall(r"[a-z0-9_]{3,}", normalized)
    stopwords = {
        "como",
        "para",
        "partir",
        "por",
        "con",
        "los",
        "las",
        "una",
        "uno",
        "que",
        "del",
        "actualizar",
        "obtener",
        "obtengo",
        "hacer",
        "esta",
        "este",
    }
    filtered_terms = [term for term in terms if term not in stopwords]
    return expand_query_terms(filtered_terms)


def expand_query_terms(terms: list[str]) -> list[str]:
    expansions = {
        "subcliente": ["idsubcliente", "subclientes"],
        "subclientes": ["subcliente", "idsubcliente"],
        "expendedora": ["expendedor", "expendedoras", "expendedores"],
        "expendedoras": ["expendedora", "expendedor", "expendedores"],
        "expendedor": ["expendedora", "expendedores", "expendedoras"],
        "expendedores": ["expendedor", "expendedora", "expendedoras"],
    }
    expanded: list[str] = []
    seen: set[str] = set()

    for term in terms:
        related_terms = [term, *expansions.get(term, [])]
        for related_term in related_terms:
            if related_term in seen:
                continue
            seen.add(related_term)
            expanded.append(related_term)

    return expanded


def lexical_search_chunks(db: Session, workspace_id: int, question: str, limit: int) -> list[dict]:
    terms = query_terms(question)
    if not terms:
        return []

    phrase = normalize_for_search(question)
    candidates = (
        db.query(Chunk)
        .join(Document)
        .filter(Document.workspace_id == workspace_id)
        .all()
    )
    normalized_candidates: list[tuple[Chunk, str]] = []
    document_frequency = {term: 0 for term in terms}

    for chunk in candidates:
        document = chunk.document
        if document is None or document.workspace is None:
            logger.warning("Skipping orphan chunk during lexical search chunk_id=%s", chunk.id)
            continue
        haystack = normalize_for_search(
            f"{document.name}\n{document.source}\n{document.file_type}\n{chunk.content}"
        )
        normalized_candidates.append((chunk, haystack))
        for term in terms:
            if term in haystack:
                document_frequency[term] += 1

    scored: list[tuple[float, Chunk]] = []
    total_candidates = max(1, len(normalized_candidates))

    for chunk, haystack in normalized_candidates:
        score = 0.0
        for term in terms:
            count = haystack.count(term)
            if not count:
                continue

            rarity_weight = total_candidates / max(1, document_frequency[term])
            length_weight = 1.8 if len(term) >= 7 else 1.0
            score += (1 + count) * rarity_weight * length_weight
        if phrase and phrase in haystack:
            score += 10
        if score:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (item[0], item[1].document.created_at, item[1].id), reverse=True)
    results: list[dict] = []
    for score, chunk in scored[:limit]:
        match = chunk_to_match(chunk, lexical_score=score)
        if match is not None:
            results.append(match)
    return results


def chunk_to_match(chunk: Chunk, distance: float | None = None, lexical_score: float | None = None) -> dict | None:
    document = chunk.document
    if document is None or document.workspace is None:
        logger.warning("Skipping orphan chunk chunk_id=%s document_id=%s", chunk.id, chunk.document_id)
        return None
    return {
        "id": chunk.vector_id,
        "text": chunk.content,
        "metadata": {
            "document_id": document.id,
            "workspace_id": document.workspace_id,
            "workspace_name": document.workspace.name,
            "document_name": document.name,
            "chunk_index": chunk.chunk_index,
            "source": document.source,
            "file_type": document.file_type,
        },
        "distance": distance,
        "lexical_score": lexical_score,
    }


def match_exists_in_workspace(db: Session, match: dict, workspace_id: int) -> bool:
    metadata = match.get("metadata") or {}
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index")
    if document_id is None or chunk_index is None:
        return False

    return (
        db.query(Chunk.id)
        .join(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
            Chunk.chunk_index == chunk_index,
        )
        .first()
        is not None
    )


def expand_matches_with_neighbor_chunks(db: Session, matches: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    seen: set[str] = set()

    for match in matches:
        match_id = match["id"]
        if match_id not in seen:
            seen.add(match_id)
            expanded.append(match)

        metadata = match["metadata"]
        document_id = metadata["document_id"]
        chunk_index = metadata["chunk_index"]
        neighbor_chunks = (
            db.query(Chunk)
            .filter(
                Chunk.document_id == document_id,
                Chunk.chunk_index.in_([chunk_index - 1, chunk_index + 1]),
            )
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        for neighbor_chunk in neighbor_chunks:
            if neighbor_chunk.vector_id in seen:
                continue
            seen.add(neighbor_chunk.vector_id)
            neighbor_match = chunk_to_match(neighbor_chunk)
            if neighbor_match is not None:
                expanded.append(neighbor_match)

    return expanded


def ingest_document(
    db: Session,
    workspace_id: int,
    name: str,
    file_type: str,
    source: str,
    content: str,
) -> Document:
    settings = get_settings()
    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        raise ValueError("Workspace no encontrado.")

    content = clean_text(content)
    if not content:
        raise ValueError("No se pudo extraer texto del documento.")

    digest = content_digest(content)
    duplicate = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id, Document.content_hash == digest)
        .first()
    )
    if duplicate:
        raise DuplicateDocumentError(
            f"Documento repetido en este workspace. Ya existe como '{duplicate.name}' (ID {duplicate.id})."
        )

    chunks = split_text(content, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No se generaron chunks a partir del contenido.")

    document = Document(
        workspace_id=workspace_id,
        name=name,
        file_type=file_type,
        source=source,
        content_hash=digest,
        content=content,
    )
    db.add(document)
    db.flush()

    vector_ids = [f"doc-{document.id}-chunk-{i}" for i in range(len(chunks))]
    embeddings = embed_texts(chunks)
    metadatas = [
        {
            "document_id": document.id,
            "workspace_id": workspace_id,
            "workspace_name": workspace.name,
            "document_name": name,
            "chunk_index": i,
            "source": source,
            "file_type": file_type,
        }
        for i in range(len(chunks))
    ]

    for i, chunk_text in enumerate(chunks):
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk_text,
                token_count=count_tokens(chunk_text),
                vector_id=vector_ids[i],
            )
        )

    get_vector_store().add_chunks(vector_ids, chunks, embeddings, metadatas)
    db.commit()
    db.refresh(document)
    return document


async def answer_question(db: Session, workspace_id: int, question: str, top_k: int | None = None) -> dict:
    settings = get_settings()
    if not db.get(Workspace, workspace_id):
        raise ValueError("Workspace no encontrado.")

    vector_store = get_vector_store()
    terms = query_terms(question)
    logger.info("RAG query workspace_id=%s top_k=%s terms=%s question=%r", workspace_id, top_k, terms, question)
    query_embedding = embed_query(question)
    result_limit = top_k or settings.default_top_k
    retrieval_limit = min(30, max(result_limit, result_limit * 4, 8))
    vector_matches = vector_store.search(query_embedding, retrieval_limit, workspace_id=workspace_id)
    lexical_matches = lexical_search_chunks(db, workspace_id, question, retrieval_limit)
    merged_matches = merge_retrieval_results(lexical_matches, vector_matches)
    valid_matches = [match for match in merged_matches if match_exists_in_workspace(db, match, workspace_id)]
    stale_matches = len(merged_matches) - len(valid_matches)
    if stale_matches:
        logger.warning("RAG skipped stale vector/db matches workspace_id=%s count=%s", workspace_id, stale_matches)
    matches = expand_matches_with_neighbor_chunks(db, valid_matches)
    logger.info(
        "RAG retrieval workspace_id=%s lexical_matches=%s vector_matches=%s valid=%s expanded=%s",
        workspace_id,
        len(lexical_matches),
        len(vector_matches),
        len(valid_matches),
        len(matches),
    )

    selected_context: list[str] = []
    sources: list[dict] = []
    used_tokens = 0

    for match in matches:
        chunk_tokens = count_tokens(match["text"])
        if used_tokens + chunk_tokens > settings.max_context_tokens:
            continue
        used_tokens += chunk_tokens
        metadata = match["metadata"]
        selected_context.append(
            f"[Fuente: {metadata['document_name']} | Chunk {metadata['chunk_index']}]\n{match['text']}"
        )
        sources.append(
            {
                "document_id": metadata["document_id"],
                "workspace_id": metadata["workspace_id"],
                "document_name": metadata["document_name"],
                "chunk_index": metadata["chunk_index"],
                "source": metadata["source"],
                "similarity_distance": match["distance"],
                "lexical_score": match.get("lexical_score"),
                "preview": match["text"][:500],
            }
        )

    if not selected_context:
        logger.info("RAG query has no selected context workspace_id=%s question=%r", workspace_id, question)
        return {
            "answer": "No hay información suficiente en los documentos cargados.",
            "sources": [],
            "context_tokens": 0,
        }

    logger.info(
        "RAG context workspace_id=%s context_tokens=%s sources=%s",
        workspace_id,
        used_tokens,
        [
            {
                "document_id": source["document_id"],
                "document_name": source["document_name"],
                "chunk_index": source["chunk_index"],
                "distance": source["similarity_distance"],
                "lexical_score": source["lexical_score"],
            }
            for source in sources
        ],
    )

    prompt = (
        "Pregunta del usuario:\n"
        f"{question}\n\n"
        "Contexto recuperado por búsqueda semántica:\n"
        f"{chr(10).join(selected_context)}\n\n"
        "Instrucciones:\n"
        "- Usá solo el contexto anterior.\n"
        "- Si el contexto incluye SQL, código o archivos relacionados con la pregunta, explicá qué muestran esos ejemplos y cómo se aplican.\n"
        "- Para preguntas tipo 'cómo hago/modifico/actualizo', respondé con pasos concretos basados en los scripts o fragmentos recuperados.\n"
        "- Si alguna parte no está en el contexto, separala claramente como límite o dato faltante.\n"
        "- Solo respondé que no hay información suficiente cuando los chunks recuperados no tengan relación útil con la pregunta.\n"
        "- No inventes datos.\n"
        "- Citá los nombres de archivos o fuentes usados cuando sean relevantes.\n"
    )
    answer = await LLMService().generate(prompt)
    return {"answer": answer, "sources": sources, "context_tokens": used_tokens}


def merge_retrieval_results(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for match in primary + secondary:
        match_id = match["id"]
        if match_id in seen:
            continue
        seen.add(match_id)
        merged.append(match)
    return merged


def delete_document(db: Session, document_id: int) -> bool:
    document = db.get(Document, document_id)
    if not document:
        return False
    get_vector_store().delete_document(document_id)
    db.delete(document)
    db.commit()
    return True


def delete_workspace(db: Session, workspace_id: int) -> bool:
    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        return False
    get_vector_store().delete_workspace(workspace_id)
    db.delete(workspace)
    db.commit()
    return True


def delete_duplicate_documents(db: Session, workspace_id: int | None = None) -> dict:
    query = db.query(Document).filter(Document.content_hash.is_not(None))
    if workspace_id is not None:
        query = query.filter(Document.workspace_id == workspace_id)

    documents = query.order_by(Document.workspace_id.asc(), Document.content_hash.asc(), Document.created_at.asc(), Document.id.asc()).all()
    seen: set[tuple[int, str]] = set()
    deleted: list[dict] = []
    vector_store = get_vector_store()

    for document in documents:
        if not document.content_hash:
            continue
        key = (document.workspace_id, document.content_hash)
        if key not in seen:
            seen.add(key)
            continue

        deleted.append(
            {
                "id": document.id,
                "name": document.name,
                "workspace_id": document.workspace_id,
                "source": document.source,
            }
        )
        vector_store.delete_document(document.id)
        db.delete(document)

    db.commit()
    return {"deleted_count": len(deleted), "deleted": deleted}
