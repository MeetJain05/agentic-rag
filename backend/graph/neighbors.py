"""
Expand relevant chunks with their neighbouring chunks.

Lists (projects, skills, steps, metrics) often straddle a chunk
boundary, so the chunk that matched the question holds only part of
the list. After grading, each relevant chunk is widened with the
chunks just before and after it in the same document, read from the
local SQLite store (no extra Pinecone or LLM calls).

Grading still runs on the original, small chunks, so this adds no
grader tokens; only generation sees the wider window.
"""

from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

from backend.config import NEIGHBOR_CHUNKS


def _chunk_index(metadata: dict) -> Optional[int]:
    try:
        return int(float(metadata.get("chunk_index")))
    except (TypeError, ValueError):
        return None


def _join_overlapping(parts: List[str], max_overlap: int = 400) -> str:
    """Concatenate chunks, dropping the overlap the splitter added."""
    if not parts:
        return ""
    text = parts[0]
    for part in parts[1:]:
        overlap = 0
        limit = min(max_overlap, len(text), len(part))
        for size in range(limit, 20, -1):
            if text.endswith(part[:size]):
                overlap = size
                break
        text = text + ("" if overlap else "\n") + part[overlap:]
    return text


def expand_with_neighbors(
    documents: List[Document],
    window: int = NEIGHBOR_CHUNKS,
) -> List[Document]:
    if window <= 0 or not documents:
        return documents

    try:
        from backend import store
    except Exception:
        return documents

    covered: Dict[str, Set[int]] = {}
    expanded: List[Document] = []

    for document in documents:
        metadata = document.metadata or {}
        document_id = str(metadata.get("document_id") or "")
        index = _chunk_index(metadata)

        if not document_id or index is None:
            expanded.append(document)
            continue

        seen = covered.setdefault(document_id, set())
        if index in seen:
            # Already included inside an earlier, higher-ranked window.
            continue

        indices = [
            i
            for i in range(index - window, index + window + 1)
            if i >= 0 and i != index and i not in seen
        ]

        try:
            texts = store.get_chunks_by_index(document_id, indices)
        except Exception as exc:
            print(f"---NEIGHBOR EXPANSION UNAVAILABLE: {exc}---")
            return documents

        texts[index] = document.page_content

        # Only use the contiguous run around the matched chunk.
        run = [index]
        for i in range(index - 1, index - window - 1, -1):
            if i not in texts:
                break
            run.insert(0, i)
        for i in range(index + 1, index + window + 1):
            if i not in texts:
                break
            run.append(i)

        seen.update(run)
        expanded.append(
            Document(
                page_content=_join_overlapping([texts[i] for i in run]),
                metadata={**metadata, "chunk_window": [run[0], run[-1]]},
            )
        )

    return expanded
