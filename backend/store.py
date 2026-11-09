"""
Local SQLite store that sits next to Pinecone.

It serves two purposes:

1. Document registry - /documents and /health read a small table
   instead of fetching every vector from Pinecone.
2. Keyword index (FTS5 / BM25) over chunk text - the lexical half of
   hybrid search. Dense vectors miss exact names, IDs and acronyms;
   BM25 catches them.

Pinecone stays the source of truth. If the SQLite file is missing
(fresh container, deleted volume) it is rebuilt from Pinecone with
`sync_from_pinecone()`, which the API runs at startup.
"""

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document

from backend.config import DATA_DIR

STORE_PATH = Path(DATA_DIR) / "rag_store.sqlite3"

_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None


def _connect() -> sqlite3.Connection:
    global _connection

    if _connection is None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(
            STORE_PATH,
            check_same_thread=False,
        )
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                file_name   TEXT NOT NULL,
                file_type   TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                metadata UNINDEXED,
                text,
                tokenize = 'porter unicode61'
            );
            """
        )
    return _connection


@contextmanager
def _transaction():
    with _lock:
        connection = _connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def add_document(
    document_id: str,
    file_name: str,
    file_type: str,
    chunks: List[Document],
    chunk_ids: List[str],
) -> None:
    """Register a document and index its chunks for keyword search."""

    now = datetime.now(timezone.utc).isoformat()

    with _transaction() as connection:
        connection.execute(
            "DELETE FROM chunks WHERE document_id = ?",
            (document_id,),
        )
        connection.executemany(
            "INSERT INTO chunks (chunk_id, document_id, metadata, text) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    chunk_id,
                    document_id,
                    json.dumps(chunk.metadata, default=str),
                    chunk.page_content,
                )
                for chunk, chunk_id in zip(chunks, chunk_ids)
            ],
        )
        connection.execute(
            "INSERT OR REPLACE INTO documents "
            "(document_id, file_name, file_type, chunk_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, file_name, file_type, len(chunks), now),
        )


def remove_document(document_id: str) -> None:
    with _transaction() as connection:
        connection.execute(
            "DELETE FROM chunks WHERE document_id = ?",
            (document_id,),
        )
        connection.execute(
            "DELETE FROM documents WHERE document_id = ?",
            (document_id,),
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_documents() -> List[Dict[str, Any]]:
    with _lock:
        rows = _connect().execute(
            "SELECT document_id, file_name, file_type, chunk_count, created_at "
            "FROM documents ORDER BY created_at DESC"
        ).fetchall()

    return [
        {
            "document_id": row["document_id"],
            "file_name": row["file_name"],
            "file_type": row["file_type"],
            "source": row["file_name"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def document_count() -> int:
    with _lock:
        return int(
            _connect().execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )


def has_document(document_id: str) -> bool:
    with _lock:
        row = _connect().execute(
            "SELECT 1 FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return row is not None


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _fts_query(text: str) -> str:
    """
    Turn free text into a safe FTS5 query.

    Each token is quoted (so punctuation/operators in user input
    can't break the query) and tokens are OR-ed; BM25 then rewards
    chunks that match more and rarer terms.
    """

    tokens = [token for token in _TOKEN.findall(text.lower()) if len(token) > 1]
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))


def keyword_search(query: str, limit: int) -> List[Document]:
    match = _fts_query(query)
    if not match:
        return []

    with _lock:
        rows = _connect().execute(
            "SELECT chunk_id, metadata, text, bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
            (match, limit),
        ).fetchall()

    documents = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        metadata["chunk_id"] = row["chunk_id"]
        # bm25() is "lower is better"; flip it so higher = better.
        metadata["keyword_score"] = -float(row["score"])
        documents.append(Document(page_content=row["text"], metadata=metadata))
    return documents


# ---------------------------------------------------------------------------
# Rebuild from Pinecone
# ---------------------------------------------------------------------------


def sync_from_pinecone(records: Iterable[Dict[str, Any]]) -> int:
    """
    Rebuild the registry and keyword index from Pinecone records
    (as yielded by backend.retrieval.iter_records). Returns the
    number of documents registered.
    """

    grouped: Dict[str, Dict[str, Any]] = {}

    for record in records:
        metadata = dict(record.get("metadata") or {})
        document_id = metadata.get("document_id")
        text = str(metadata.pop("chunk_text", "") or "")
        if not document_id or not text.strip():
            continue

        entry = grouped.setdefault(
            document_id,
            {
                "file_name": str(metadata.get("file_name") or "Unknown"),
                "file_type": str(metadata.get("file_type") or ""),
                "chunks": [],
                "ids": [],
            },
        )
        entry["chunks"].append(Document(page_content=text, metadata=metadata))
        entry["ids"].append(record["id"])

    for document_id, entry in grouped.items():
        add_document(
            document_id=document_id,
            file_name=entry["file_name"],
            file_type=entry["file_type"],
            chunks=entry["chunks"],
            chunk_ids=entry["ids"],
        )

    return len(grouped)
