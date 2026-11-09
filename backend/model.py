import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


# -----------------------------
# LLM Configuration
# -----------------------------

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-120b",
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        # Low temperature keeps RAG answers close to the evidence.
        "0.2",
    )
)

LLM_MAX_RETRIES = int(
    os.getenv(
        "LLM_MAX_RETRIES",
        "2",
    )
)


# -----------------------------
# Groq LLM
# -----------------------------

llm_model = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=LLM_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
)



# -----------------------------
# Deterministic LLM for routing / grading
# -----------------------------
#
# Yes/no judges should not be sampled at temperature 0.7: the same
# answer could pass on one run and fail on the next, which also makes
# the retry loop and evaluation metrics noisy.

GRADER_TEMPERATURE = float(
    os.getenv(
        "GRADER_TEMPERATURE",
        "0",
    )
)

grader_llm_model = ChatGroq(
    model=os.getenv("GRADER_MODEL", LLM_MODEL),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=GRADER_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
)


# -----------------------------
# Embeddings (Pinecone hosted inference)
# -----------------------------
#
# Same model as the integrated index, computed by Pinecone's API, so
# no local embedding model is loaded (important on 512 MB hosts).
# Used by RAGAS (answer_relevancy).

from typing import List

from langchain_core.embeddings import Embeddings


class PineconeInferenceEmbeddings(Embeddings):
    def __init__(self, model: str, batch_size: int = 90):
        self.model = model
        self.batch_size = batch_size

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        from backend.retrieval import get_pinecone_client

        vectors: List[List[float]] = []
        client = get_pinecone_client()
        for start in range(0, len(texts), self.batch_size):
            result = client.inference.embed(
                model=self.model,
                inputs=texts[start:start + self.batch_size],
                parameters={"input_type": input_type, "truncate": "END"},
            )
            for item in result.data:
                values = getattr(item, "values", None)
                if values is None and isinstance(item, dict):
                    values = item.get("values")
                vectors.append(list(values or []))
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]


embed_model = PineconeInferenceEmbeddings(
    model=os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large"),
)
