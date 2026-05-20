import re

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """Approximate prompt size. Embeddings are for similarity; tokens limit LLM context."""

    if tiktoken:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    return max(1, len(text.split()))


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping word chunks."""

    text = clean_text(text)
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks
