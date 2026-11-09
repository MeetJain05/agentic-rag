from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.model import grader_llm_model as llm_model


system = """
You rewrite a user's question into a better search query for a
document retrieval system (semantic + keyword search).

The previous search did not return evidence that fully supported an
answer. Produce ONE improved query that:

- keeps the original intent and every named entity, number or code;
- expands abbreviations and adds likely synonyms or related terms;
- removes conversational filler.

Return only the rewritten query - no quotes, no explanation.
"""


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Original question:\n{question}\n\n"
            "Previous search query:\n{previous_query}\n\n"
            "Improved search query:",
        ),
    ]
)


question_rewriter = rewrite_prompt | llm_model | StrOutputParser()
