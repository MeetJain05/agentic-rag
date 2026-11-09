from langchain_core.documents import Document

from backend.graph.neighbors import _join_overlapping, expand_with_neighbors
from backend.graph.nodes.greeting import is_greeting
from backend.graph.nodes.security import route_after_security
from backend.observability import calculate_observability_summary


def test_pure_greetings_are_detected() -> None:
    for text in [
        "hi",
        "Hello there 👋",
        "hey, how are you?",
        "good morning",
        "Thanks a lot!",
        "bye",
        "hi and thanks",
    ]:
        assert is_greeting(text), text


def test_questions_are_not_greetings() -> None:
    for text in [
        "hi, what models were used?",
        "what is AI",
        "thanks, now list the metrics",
        "hey give me all three projects",
        "how are you trained",
        "ok",
    ]:
        assert not is_greeting(text), text


def test_security_routes_greetings_but_not_blocked_input() -> None:
    assert route_after_security({"question": "hi"}) == "greeting"
    assert route_after_security({"question": "what is RAG?"}) == "allowed"
    assert (
        route_after_security({"question": "hi", "security_status": "blocked"})
        == "blocked"
    )


def test_route_counts_use_recorded_route_names() -> None:
    summary = calculate_observability_summary(
        [
            {"route": "local", "grounded": True, "answers_question": True, "success": True},
            {"route": "web", "hitl_status": "pending", "success": True},
            {"route": "web", "hitl_status": "approved", "grounded": True, "answers_question": True, "success": True},
            {"route": "web_rejected", "hitl_status": "rejected", "success": True},
            {"route": "greeting", "grounded": True, "answers_question": True, "success": True},
        ]
    )
    assert summary["local_route_count"] == 1
    assert summary["web_route_count"] == 2  # pending entry not double-counted
    assert summary["hitl_approved_count"] == 1
    assert summary["hitl_rejected_count"] == 1
    # Greeting and pending runs are excluded: 2 of 3 RAG runs grounded.
    assert summary["grounded_rate"] == round(2 / 3, 4)


def test_join_overlapping_removes_splitter_overlap() -> None:
    a = "intro " + "x" * 30 + " shared overlap segment here"
    b = "shared overlap segment here and the rest"
    assert _join_overlapping([a, b]).endswith("here and the rest")
    assert _join_overlapping([a, b]).count("shared overlap") == 1


def test_expansion_merges_adjacent_hits(monkeypatch) -> None:
    from backend import store

    chunks = {i: f"chunk {i} text" for i in range(6)}
    monkeypatch.setattr(
        store,
        "get_chunks_by_index",
        lambda document_id, indices: {i: chunks[i] for i in indices if i in chunks},
    )
    docs = [
        Document(page_content=chunks[2], metadata={"document_id": "d", "chunk_index": 2.0}),
        Document(page_content=chunks[3], metadata={"document_id": "d", "chunk_index": 3}),
        Document(page_content="web", metadata={"source": "web"}),
    ]
    expanded = expand_with_neighbors(docs, window=1)
    assert [d.metadata.get("chunk_window") for d in expanded] == [[1, 3], None]
    assert "chunk 1" in expanded[0].page_content and "chunk 3" in expanded[0].page_content
