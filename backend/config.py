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
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

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