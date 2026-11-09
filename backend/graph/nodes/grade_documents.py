from typing import Any, Dict

from backend.graph.chains.retrieval_grader import retrieval_grader
from backend.graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grade retrieved documents for relevance to the question.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]

    documents = state.get(
        "documents",
        [],
    )

    filtered_docs = []

    # Grade all chunks concurrently instead of one LLM call at a time.
    scores = retrieval_grader.batch(
        [
            {
                "question": question,
                "document": document.page_content,
            }
            for document in documents
        ],
        config={"max_concurrency": 4},
    ) if documents else []

    for document, score in zip(documents, scores):

        if str(score.binary_score).strip().lower() == "yes":

            print(
                "---GRADE: DOCUMENT RELEVANT---"
            )

            filtered_docs.append(document)

        else:

            print(
                "---GRADE: DOCUMENT NOT RELEVANT---"
            )

    relevant_documents = len(filtered_docs)

    web_search = relevant_documents == 0

    if web_search:

        print(
            "---NO RELEVANT DOCUMENTS FOUND---"
        )

        print(
            "---WEB SEARCH REQUIRED---"
        )

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "relevant_documents": relevant_documents,
    }