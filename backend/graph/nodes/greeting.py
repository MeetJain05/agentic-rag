"""
Greeting / pleasantry shortcut.

A message that is *only* a greeting, thanks or goodbye ("hi",
"hello there!", "good morning", "thanks a lot", "bye") is answered
directly by the LLM, skipping retrieval, routing, web-search approval
and the graders.

Detection is a strict, deterministic match on the whole message, not
an LLM classifier, so it never captures real questions:

- "hi"                          -> greeting
- "hey, how are you?"           -> greeting
- "hi, what models were used?"  -> normal RAG pipeline
- "what is AI"                  -> normal RAG pipeline
"""

import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.graph.state import GraphState
from backend.model import llm_model

MAX_GREETING_CHARS = 60

_HELLO = (
    r"h+i+|h+e+l+o+|hey+|heya|hiya|yo|howdy|greetings|namaste|hola"
    r"|good (?:morning|afternoon|evening|day)"
    r"|how are you(?: doing)?|how(?: i|')?s it going|how do you do"
    r"|what(?: i|')?s up|sup|nice to meet you"
)
_THANKS = (
    r"thanks?|thank you|thx|ty|tysm|much appreciated|appreciate it"
)
_BYE = (
    r"bye+|goodbye|good bye|see (?:you|ya)(?: later)?|cya"
    r"|good night|take care|have a (?:good|nice|great) (?:day|one)"
)
# Words that may follow a greeting ("hi there", "thanks so much").
_FILLER = (
    r"there|all|everyone|again|so much|a lot|very much|buddy|friend"
    r"|bot|assistant|agent|team|mate|man|today|today\?"
)

_PHRASE = rf"(?:{_HELLO}|{_THANKS}|{_BYE})(?:\s+(?:{_FILLER}))*"
_GREETING_RE = re.compile(rf"^{_PHRASE}(?:\s+(?:and\s+)?{_PHRASE}){{0,2}}$")
_THANKS_RE = re.compile(rf"^(?:{_THANKS})\b")
_BYE_RE = re.compile(rf"^(?:{_BYE})\b")


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    # Drop punctuation and emoji; keep letters, digits, spaces, apostrophes.
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def is_greeting(text: str) -> bool:
    if not isinstance(text, str) or len(text.strip()) > MAX_GREETING_CHARS:
        return False
    normalised = _normalise(text)
    return bool(normalised) and bool(_GREETING_RE.match(normalised))


def _fallback_reply(text: str) -> str:
    normalised = _normalise(text)
    if _THANKS_RE.match(normalised):
        return "You're welcome! Ask me anything else about your documents."
    if _BYE_RE.match(normalised):
        return "Goodbye! Come back any time you have questions."
    return (
        "Hi! I can answer questions about the documents you've "
        "uploaded. What would you like to know?"
    )


_greeting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the assistant in a document question-answering app. The user
has sent only a greeting, thanks or goodbye.

Reply in one or two short, friendly sentences that match their message:

- greeting: greet them back and offer to help with questions about
  their uploaded documents;
- thanks: say you're welcome and invite another question;
- goodbye: say goodbye politely.

Do not mention or invent anything about the documents' contents, do
not ask more than one question, and do not use headings or lists.
""",
        ),
        ("human", "{question}"),
    ]
)

_greeting_chain = _greeting_prompt | llm_model | StrOutputParser()


def respond_greeting(state: GraphState) -> Dict[str, Any]:
    print("---GREETING: SKIPPING RETRIEVAL---")

    question = state["question"]

    try:
        reply = " ".join(str(_greeting_chain.invoke({"question": question})).split())
    except Exception as exc:
        print(f"---GREETING LLM FAILED: {exc}---")
        reply = ""

    reply = reply or _fallback_reply(question)

    return {
        "question": question,
        "generation": reply,
        "route": "greeting",
        "documents": [],
        "retrieved_documents": 0,
        "relevant_documents": 0,
        # Not applicable to small talk; the UI and metrics skip them.
        "grounded": True,
        "answers_question": True,
    }
