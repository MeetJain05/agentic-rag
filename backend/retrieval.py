import os
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document
from pinecone import Pinecone

from backend.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RETRIEVAL_K,
)


def get_pinecone_index():
    api_key = PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is required to use the Pinecone vector store."
        )

    return Pinecone(api_key=api_key).Index(PINECONE_INDEX_NAME)


def _vector_ids(batch: Iterable[Any]) -> List[str]:
    ids = []
    for item in batch:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif getattr(item, "id", None):
            ids.append(str(item.id))
    return ids


def iter_records() -> Iterable[Dict[str, Any]]:
    index = get_pinecone_index()
    for batch in index.list(namespace=PINECONE_NAMESPACE):
        ids = _vector_ids(batch)
        if not ids:
            continue

        response = index.fetch(
            ids=ids,
            namespace=PINECONE_NAMESPACE,
        )
        vectors = getattr(response, "vectors", {})
        if not isinstance(vectors, dict):
            continue

        for vector_id, vector in vectors.items():
            metadata = getattr(vector, "metadata", {}) or {}
            yield {
                "id": vector_id,
                "metadata": metadata,
            }


def upsert_documents(
    documents: List[Document],
    ids: List[str],
) -> None:
    if len(documents) != len(ids):
        raise ValueError("Each document must have exactly one vector ID.")

    records = []
    for document, vector_id in zip(documents, ids):
        records.append(
            {
                "_id": vector_id,
                "chunk_text": document.page_content,
                **document.metadata,
            }
        )

    get_pinecone_index().upsert_records(
        namespace=PINECONE_NAMESPACE,
        records=records,
    )


class PineconeIntegratedRetriever:
    def invoke(self, query: str) -> List[Document]:
        response = get_pinecone_index().search(
            namespace=PINECONE_NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": RETRIEVAL_K,
            },
            fields=["*"],
        )
        if hasattr(response, "to_dict"):
            response_data = response.to_dict()
        elif isinstance(response, dict):
            response_data = response
        else:
            response_data = {}

        result = response_data.get("result", {})
        hits = result.get("hits", []) if isinstance(result, dict) else []

        documents = []
        for hit in hits:
            fields = hit.get("fields", {})
            if not isinstance(fields, dict):
                continue

            content = str(fields.get("chunk_text", "")).strip()
            if not content:
                continue

            metadata = {
                key: value
                for key, value in fields.items()
                if key != "chunk_text"
            }
            metadata["pinecone_score"] = hit.get(
                "_score",
                hit.get("score_", 0.0),
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        return documents


retriever = PineconeIntegratedRetriever()


def has_documents() -> bool:
    return next(iter(iter_records()), None) is not None
