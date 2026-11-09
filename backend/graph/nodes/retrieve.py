from typing import Any, Dict

from backend.graph.state import GraphState
from backend.retrieval import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents for the RAG pipeline.

    - After a query rewrite (retry path), search again with the
      rewritten query.
    - Otherwise reuse the documents fetched during evidence-aware
      routing, even if empty: searching the same index again for the
      same question would be wasted work.

    Transient provider errors are retried inside the retriever.
    """

    print("---RETRIEVE---")

    question = state["question"]
    search_query = state.get("search_query")

    if search_query:
        documents = retriever.invoke(search_query)
        print(
            f"---RETRIEVED {len(documents)} DOCUMENTS "
            f"FOR REWRITTEN QUERY---"
        )
        return {
            "documents": documents,
            "question": question,
            "retrieved_documents": len(documents),
        }

    if "candidate_documents" in state:
        candidate_documents = state.get("candidate_documents") or []
        print(
            "---REUSING ROUTER RETRIEVAL: "
            f"{len(candidate_documents)} DOCUMENTS---"
        )
        return {
            "documents": candidate_documents,
            "question": question,
            "retrieved_documents": len(candidate_documents),
        }

    documents = retriever.invoke(question)
    print(f"---RETRIEVED {len(documents)} DOCUMENTS---")

    return {
        "documents": documents,
        "question": question,
        "retrieved_documents": len(documents),
    }
