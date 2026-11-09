from typing import Any, Dict

from backend.graph.chains.question_rewriter import question_rewriter
from backend.graph.state import GraphState


MAX_QUERY_CHARS = 500


def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Rewrite the search query before a retry.

    When an answer is not grounded or doesn't answer the question,
    the usual culprit is retrieval, so regenerating from the same
    evidence rarely helps. The original question is kept for
    grading/generation; only the search query changes.
    """

    print("---TRANSFORM QUERY---")

    question = state["question"]
    previous_query = state.get("search_query") or question

    try:
        rewritten = question_rewriter.invoke(
            {
                "question": question,
                "previous_query": previous_query,
            }
        )
        rewritten = " ".join(str(rewritten).split())[:MAX_QUERY_CHARS]
    except Exception as exc:
        print(f"---QUERY REWRITE FAILED: {exc}---")
        rewritten = ""

    if not rewritten:
        rewritten = previous_query

    print(f"---REWRITTEN QUERY: {rewritten}---")

    return {
        "search_query": rewritten,
    }
