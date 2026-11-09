import os
import time
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document
from pinecone import Pinecone

from backend.config import (
    HYBRID_SEARCH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_RERANK_MODEL,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_K,
)


# Pinecone integrated-embedding indexes accept at most 96 text
# records per upsert request. Larger PDFs easily exceed this.
UPSERT_BATCH_SIZE = 96

MAX_SEARCH_RETRIES = 2
SEARCH_RETRY_DELAY_SECONDS = 1.5

# Reciprocal rank fusion constant (standard value from the RRF paper).
RRF_K = 60

_client = None
_index = None


def get_pinecone_client() -> Pinecone:
    """Return a cached Pinecone client."""

    global _client

    if _client is None:
        api_key = PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is required to use the Pinecone vector store."
            )

        _client = Pinecone(api_key=api_key)
    return _client


def get_pinecone_index():
    """Return a cached Pinecone index handle."""

    global _index

    if _index is None:
        _index = get_pinecone_client().Index(PINECONE_INDEX_NAME)
    return _index


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


def list_ids(prefix: str = "") -> List[str]:
    """
    List vector IDs in the namespace, optionally filtered by ID prefix.

    Chunk IDs start with the document's content hash, so a prefix
    lookup touches only that document's vectors instead of the
    whole index.
    """

    kwargs: Dict[str, Any] = {"namespace": PINECONE_NAMESPACE}
    if prefix:
        kwargs["prefix"] = prefix

    ids: List[str] = []
    for batch in get_pinecone_index().list(**kwargs):
        ids.extend(_vector_ids(batch))
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


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pinecone metadata accepts only str, number, bool or list[str].
    Null values (common in PDF metadata) make the whole upsert fail.
    """

    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or key == "chunk_text":
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [str(item) for item in value]
        else:
            cleaned[key] = str(value)
    return cleaned


def upsert_documents(
    documents: List[Document],
    ids: List[str],
) -> None:
    if len(documents) != len(ids):
        raise ValueError("Each document must have exactly one vector ID.")

    records = [
        {
            "_id": vector_id,
            "chunk_text": document.page_content,
            **_clean_metadata(document.metadata),
        }
        for document, vector_id in zip(documents, ids)
    ]

    index = get_pinecone_index()
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert_records(
            namespace=PINECONE_NAMESPACE,
            records=records[start:start + UPSERT_BATCH_SIZE],
        )


def _response_hits(response: Any) -> List[Dict[str, Any]]:
    if hasattr(response, "to_dict"):
        response_data = response.to_dict()
    elif isinstance(response, dict):
        response_data = response
    else:
        response_data = {}

    result = response_data.get("result", {})
    return result.get("hits", []) if isinstance(result, dict) else []


def _hit_to_document(hit: Dict[str, Any]) -> Document | None:
    fields = hit.get("fields", {})
    if not isinstance(fields, dict):
        return None

    content = str(fields.get("chunk_text", "")).strip()
    if not content:
        return None

    metadata = {
        key: value
        for key, value in fields.items()
        if key != "chunk_text"
    }
    metadata["chunk_id"] = hit.get("_id", hit.get("id", ""))
    metadata["pinecone_score"] = hit.get(
        "_score",
        hit.get("score_", 0.0),
    )
    return Document(page_content=content, metadata=metadata)


def _doc_key(document: Document) -> str:
    return str(document.metadata.get("chunk_id") or document.page_content[:200])


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """
    Merge ranked lists without needing comparable scores:
    score(d) = sum over lists of 1 / (k + rank_in_list).
    """

    scores: Dict[str, float] = {}
    documents: Dict[str, Document] = {}

    for results in result_lists:
        for rank, document in enumerate(results, start=1):
            key = _doc_key(document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key in documents:
                # Keep the scores from both retrievers for observability.
                documents[key].metadata.update(
                    {
                        name: value
                        for name, value in document.metadata.items()
                        if name.endswith("_score")
                    }
                )
            else:
                documents[key] = document

    ordered = sorted(scores, key=scores.get, reverse=True)
    for key in ordered:
        documents[key].metadata["rrf_score"] = round(scores[key], 5)
    return [documents[key] for key in ordered]


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int,
) -> List[Document]:
    """Rerank with a Pinecone-hosted cross-encoder; fall back on error."""

    if not PINECONE_RERANK_MODEL or len(documents) <= 1:
        return documents[:top_n]

    try:
        result = get_pinecone_client().inference.rerank(
            model=PINECONE_RERANK_MODEL,
            query=query,
            documents=[
                {"id": str(i), "text": document.page_content}
                for i, document in enumerate(documents)
            ],
            rank_fields=["text"],
            top_n=min(top_n, len(documents)),
            return_documents=False,
            parameters={"truncate": "END"},
        )
    except Exception as exc:
        print(f"---RERANK FAILED, USING FUSED ORDER: {exc}---")
        return documents[:top_n]

    reranked = []
    for row in getattr(result, "data", []) or []:
        index = getattr(row, "index", None)
        if index is None and isinstance(row, dict):
            index = row.get("index")
        if index is None or not 0 <= int(index) < len(documents):
            continue
        document = documents[int(index)]
        score = getattr(row, "score", None)
        if score is None and isinstance(row, dict):
            score = row.get("score")
        document.metadata["rerank_score"] = score
        reranked.append(document)

    return reranked or documents[:top_n]


class HybridRetriever:
    """
    Dense (Pinecone integrated embeddings) + keyword (BM25 via SQLite
    FTS5) retrieval, fused with reciprocal rank fusion and reranked
    by a Pinecone-hosted reranker.

    Each stage degrades gracefully: no keyword index -> dense only,
    no/failed reranker -> fused order.
    """

    def _dense_search(self, query: str, top_k: int) -> List[Document]:
        last_error: Exception | None = None

        for attempt in range(MAX_SEARCH_RETRIES + 1):
            try:
                response = get_pinecone_index().search(
                    namespace=PINECONE_NAMESPACE,
                    query={
                        "inputs": {"text": query},
                        "top_k": top_k,
                    },
                    fields=["*"],
                )
                break
            except Exception as exc:  # transient provider errors
                last_error = exc
                if attempt >= MAX_SEARCH_RETRIES:
                    raise
                print(
                    f"---RETRIEVAL ERROR: {exc} "
                    f"(retry {attempt + 1}/{MAX_SEARCH_RETRIES})---"
                )
                time.sleep(SEARCH_RETRY_DELAY_SECONDS)
        else:  # pragma: no cover - loop always breaks or raises
            raise last_error or RuntimeError("Retrieval failed.")

        documents = []
        for hit in _response_hits(response):
            document = _hit_to_document(hit)
            if document is not None:
                documents.append(document)
        return documents

    def _keyword_search(self, query: str, top_k: int) -> List[Document]:
        if not HYBRID_SEARCH:
            return []
        try:
            from backend.store import keyword_search

            return keyword_search(query, top_k)
        except Exception as exc:
            print(f"---KEYWORD SEARCH UNAVAILABLE: {exc}---")
            return []

    def invoke(self, query: str) -> List[Document]:
        widen = bool(PINECONE_RERANK_MODEL) or HYBRID_SEARCH
        candidates = (
            max(RETRIEVAL_CANDIDATES, RETRIEVAL_K) if widen else RETRIEVAL_K
        )

        dense = self._dense_search(query, candidates)
        keyword = self._keyword_search(query, candidates)

        fused = reciprocal_rank_fusion([dense, keyword]) if keyword else dense

        return rerank_documents(query, fused, RETRIEVAL_K)


# Backwards-compatible name.
PineconeIntegratedRetriever = HybridRetriever


retriever = HybridRetriever()


def has_documents() -> bool:
    for batch in get_pinecone_index().list(namespace=PINECONE_NAMESPACE):
        if _vector_ids(batch):
            return True
    return False
