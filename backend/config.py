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