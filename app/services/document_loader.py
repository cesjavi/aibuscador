import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from docx import Document as DocxDocument
from fastapi import UploadFile
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls", ".cs", ".sql"}


def iter_supported_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"La carpeta no existe o no es válida: {folder}")

    ignored_dirs = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    files: list[Path] = []

    def scan(directory: Path) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return

        for entry in entries:
            try:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if path.name.lower() not in ignored_dirs:
                        scan(path)
                    continue
                if entry.is_file(follow_symlinks=False) and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(path)
            except OSError:
                continue

    scan(folder)
    return sorted(files)


def extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in {".txt", ".cs", ".sql"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        return _extract_dataframe(pd.read_csv(path))
    if suffix in {".xlsx", ".xls"}:
        return _extract_dataframe(pd.read_excel(path))
    raise ValueError(f"Tipo de archivo no soportado: {suffix}")


async def save_upload(upload: UploadFile, upload_dir: Path) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Tipo de archivo no soportado: {suffix}")

    safe_name = Path(upload.filename or f"upload{suffix}").name
    destination = upload_dir / safe_name

    counter = 1
    while destination.exists():
        destination = upload_dir / f"{destination.stem}_{counter}{suffix}"
        counter += 1

    content = await upload.read()
    destination.write_bytes(content)
    return destination


async def extract_text_from_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Tipo de archivo no soportado: {suffix}")

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        tmp_path = Path(tmp.name)
    try:
        return extract_text_from_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _extract_dataframe(df: pd.DataFrame) -> str:
    return df.fillna("").to_csv(index=False)
