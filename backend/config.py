import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-adaptive-rag-integrated",
)
PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "default",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
# A zero overlap cuts sentences/definitions in half at chunk
# boundaries; ~15% overlap is a safer default.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# After grading, widen each relevant chunk with this many chunks on
# either side (same document) so lists split across chunks stay whole.
NEIGHBOR_CHUNKS = int(os.getenv("NEIGHBOR_CHUNKS", "1"))

# Reranking: RETRIEVAL_CANDIDATES chunks are gathered (dense + keyword)
# and reranked down to RETRIEVAL_K with a Pinecone-hosted reranker.
# Set PINECONE_RERANK_MODEL to an empty string to disable.
PINECONE_RERANK_MODEL = os.getenv(
    "PINECONE_RERANK_MODEL",
    "bge-reranker-v2-m3",
).strip()
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "12"))

# Hybrid search: add BM25 keyword hits (local SQLite FTS5) to the
# dense Pinecone hits, merged with reciprocal rank fusion.
HYBRID_SEARCH = (
    os.getenv("HYBRID_SEARCH", "true").lower() in {"true", "1", "yes"}
)

# Embedding model of the integrated Pinecone index (used for RAGAS).
PINECONE_EMBED_MODEL = os.getenv(
    "PINECONE_EMBED_MODEL",
    "multilingual-e5-large",
)

def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"true", "1", "yes"}


# PDF parsing: "pymupdf" (Markdown with headings/lists) or "pypdf"
# (plain text). Falls back to pypdf if PyMuPDF is missing or fails.
#
# Defaults are sized for small hosts (Render free tier: 512 MB, ~0.1 CPU).
# Measured on a 60-page PDF:
#   pymupdf, layout off, tables off   ~65 MB peak,   ~3 s   (default)
#   pymupdf, tables on                ~71 MB peak,  ~20 s
#   pymupdf, layout model on         ~316 MB peak,  ~38 s
#   pypdf                             ~84 MB peak,   ~3 s
PDF_PARSER = os.getenv("PDF_PARSER", "pymupdf").strip().lower()
PDF_TABLES = _flag("PDF_TABLES", "false")
PDF_LAYOUT_MODEL = _flag("PDF_LAYOUT_MODEL", "false")

# OCR for image-only PDF pages. CPU-heavy and needs the Tesseract
# binary (Docker: apt-get install tesseract-ocr), so it is opt-in.
PDF_OCR = _flag("PDF_OCR", "false")
PDF_OCR_LANGUAGE = os.getenv("PDF_OCR_LANGUAGE", "eng")

# Local state: document registry, keyword index, HITL checkpoints.
DATA_DIR = os.getenv("DATA_DIR", "./data")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

LANGSMITH_TRACING = (
    os.getenv("LANGSMITH_TRACING", "false").lower()
    in {"true", "1", "yes"}
)

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "Agentic-Adaptive-RAG",
)