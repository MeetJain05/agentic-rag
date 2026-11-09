from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from backend.model import grader_llm_model as llm_model


llm = llm_model


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Whether the answer addresses the user's question."
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)


system = """
You grade whether an AI answer addresses the user's question.

Return true when the answer:

- directly responds to what was asked; or
- gives the information that is available and clearly states which
  part could not be found (an honest partial answer counts).

Return false when the answer is off-topic, evades the question, only
restates it, or refuses without saying what is missing.

Judge relevance and completeness only - not style, length or
formatting, and not factual accuracy (that is checked separately).
"""


answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "AI answer:\n<answer>\n{generation}\n</answer>",
        ),
    ]
)


answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
