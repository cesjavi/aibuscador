from collections.abc import Generator
from hashlib import sha256

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database.models import Base, Workspace


settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_default_workspace()
    _migrate_legacy_documents()


def _ensure_default_workspace() -> None:
    with SessionLocal() as db:
        existing = db.query(Workspace).filter(Workspace.name == "Default").first()
        if not existing:
            db.add(Workspace(name="Default", description="Workspace inicial"))
            db.commit()


def _migrate_legacy_documents() -> None:
    """Add workspace_id to older SQLite databases created before workspaces existed."""

    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "workspace_id" in columns:
        _migrate_content_hash(columns)
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE documents ADD COLUMN workspace_id INTEGER"))
        connection.execute(
            text(
                "UPDATE documents SET workspace_id = "
                "(SELECT id FROM workspaces WHERE name = 'Default' LIMIT 1)"
            )
        )
    _migrate_content_hash(columns)


def _migrate_content_hash(columns: set[str]) -> None:
    if "content_hash" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64)"))

    with engine.begin() as connection:
        rows = connection.execute(
            text("SELECT id, content FROM documents WHERE content_hash IS NULL OR content_hash = ''")
        ).mappings()
        for row in rows:
            digest = sha256((row["content"] or "").encode("utf-8")).hexdigest()
            connection.execute(
                text("UPDATE documents SET content_hash = :digest WHERE id = :id"),
                {"digest": digest, "id": row["id"]},
            )


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
