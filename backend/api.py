from __future__ import annotations

import io
import uuid

import json
import zipfile
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from backend.evaluation import EvaluationResult, EvaluationTracker
from backend.ingestion import (
    SUPPORTED_EXTENSIONS,
    build_vectorstore,
    delete_document,
    list_documents,
    sync_registry_from_pinecone,
)
from backend import store
from backend.retrieval import get_pinecone_index

from backend.observability import (
    RunObserver,
    OBSERVABILITY_FILE,
    load_observability_history,
    calculate_observability_summary,
    clear_observability_history,
)

from backend.graph.graph import app as rag_app
from backend.graph.graph import checkpointer as rag_checkpointer


def forget_thread(thread_id: str) -> None:
    """
    Drop checkpoints for a finished run. Only runs paused for HITL
    approval need saved state; keeping every finished thread would
    grow the checkpoint DB forever (bad on free-tier Postgres).
    """

    try:
        rag_checkpointer.delete_thread(thread_id)
    except Exception as exc:
        print(f"---CHECKPOINT CLEANUP FAILED: {exc}---")


load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000",
)

MAX_UPLOAD_SIZE_MB = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "25")
)

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)

# ============================================================================
# FASTAPI
# ============================================================================

api = FastAPI(
    title="Agentic Adaptive RAG API",
    description=(
        "Agentic RAG backend with adaptive routing, "
        "document ingestion, HITL web-search approval, "
        "observability, security and evaluation."
    ),
    version="1.0.0",
)

# Alias retained for compatibility.
app = api


api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=(
        r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+)"
        r"(:\d+)?$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class HITLRequest(BaseModel):
    thread_id: str
    approved: bool


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def safe_str(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def normalise_route(
    result: Dict[str, Any],
) -> str:

    route = result.get("route")

    if route:

        route = str(route)

        if route == "vectorstore":
            return "local"

        if route == "websearch":
            return "web"

        return route

    if result.get("web_search") is True:
        return "web"

    return "unknown"


def get_thread_id(
    thread_id: Optional[str],
) -> str:

    if thread_id:
        return thread_id

    return "default"


# ============================================================================
# EVALUATION HELPERS
# ============================================================================


def build_evaluation_result(
    question: str,
    result: Dict[str, Any],
    latency_seconds: float,
    error: str = "",
) -> EvaluationResult:

    answer = safe_str(
        result.get(
            "answer",
            result.get(
                "generation",
                "",
            ),
        )
    )

    documents = result.get(
        "documents",
        [],
    ) or []

    retrieved_documents = safe_int(
        result.get(
            "retrieved_documents",
            len(documents),
        )
    )

    relevant_documents = safe_int(
        result.get(
            "relevant_documents",
            0,
        )
    )

    return EvaluationResult(
        question=question,
        answer=answer,
        route=normalise_route(result),
        retrieved_documents=retrieved_documents,
        relevant_documents=relevant_documents,
        grounded=safe_bool(
            result.get(
                "grounded",
                False,
            )
        ),
        answers_question=safe_bool(
            result.get(
                "answers_question",
                False,
            )
        ),
        retry_count=safe_int(
            result.get(
                "retry_count",
                0,
            )
        ),
        latency_seconds=round(
            latency_seconds,
            4,
        ),
        sources=result.get(
            "sources",
            [],
        )
        or [],
        error=error,
    )


def record_evaluation(
    question: str,
    result: Dict[str, Any],
    latency_seconds: float,
    error: str = "",
) -> None:

    try:

        tracker = EvaluationTracker()

        evaluation_result = (
            build_evaluation_result(
                question=question,
                result=result,
                latency_seconds=latency_seconds,
                error=error,
            )
        )

        tracker.add(
            evaluation_result
        )

    except Exception as exc:

        # Evaluation must never break the
        # actual RAG application.
        print(
            "WARNING: Evaluation recording failed:",
            str(exc),
        )


# ============================================================================
# ROOT / HEALTH
# ============================================================================


@api.get("/")
def root() -> Dict[str, str]:

    return {
        "name": "Agentic Adaptive RAG API",
        "status": "running",
    }


@api.on_event("startup")
def ensure_document_registry() -> None:
    """
    Rebuild the local registry/keyword index from Pinecone when it is
    empty (first run, new container, ephemeral disk such as Render's
    free tier, which is wiped on every restart/spin-down).

    Runs in a background thread so the server binds its port and
    answers health checks immediately; uploads stay safe meanwhile
    because duplicate detection falls back to Pinecone.
    """

    def _sync() -> None:
        try:
            if store.document_count() == 0:
                synced = sync_registry_from_pinecone()
                print(f"---REGISTRY SYNCED FROM PINECONE: {synced} DOCUMENTS---")
        except Exception as exc:
            # Startup must not fail if Pinecone is briefly unreachable.
            print(f"---REGISTRY SYNC SKIPPED: {exc}---")

    import threading

    threading.Thread(target=_sync, name="registry-sync", daemon=True).start()


@api.get("/health")
def health() -> Dict[str, Any]:

    try:
        # One cheap stats call proves Pinecone is reachable; the
        # document count comes from the local registry.
        get_pinecone_index().describe_index_stats()

        return {
            "status": "healthy",
            "documents": store.document_count(),
            "evaluation_enabled": True,
            "observability_enabled": True,
        }

    except Exception as exc:

        return {
            "status": "degraded",
            "error": str(exc),
            "evaluation_enabled": True,
            "observability_enabled": True,
        }


# ============================================================================
# DOCUMENT MANAGEMENT
# ============================================================================


def validate_uploaded_content(
    filename: str,
    contents: bytes,
) -> tuple[bool, str]:
    """
    Validate uploaded content using the file extension and
    basic content signatures/structure.

    This is intentionally conservative. It is not a malware
    scanner; it prevents obvious extension/content mismatches
    and malformed supported document types.
    """

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported file type: "
            f"{extension or 'unknown'}"
        )

    if extension == ".pdf":
        if not contents.startswith(b"%PDF-"):
            return False, (
                "File does not contain a valid PDF signature."
            )

        return True, ""

    if extension == ".docx":
        if not contents.startswith(b"PK"):
            return False, (
                "File does not contain a valid DOCX/ZIP signature."
            )

        try:
            with zipfile.ZipFile(
                io.BytesIO(contents)
            ) as archive:
                names = set(archive.namelist())

                required_files = {
                    "[Content_Types].xml",
                    "word/document.xml",
                }

                if not required_files.issubset(names):
                    return False, (
                        "File does not contain the required DOCX structure."
                    )

        except zipfile.BadZipFile:
            return False, (
                "File is not a valid DOCX archive."
            )

        return True, ""

    if extension in {".txt", ".md"}:
        if b"\x00" in contents:
            return False, (
                "Text file contains binary data."
            )

        try:
            contents.decode("utf-8")
        except UnicodeDecodeError:
            return False, (
                "Text file is not valid UTF-8."
            )

        return True, ""

    return False, (
        f"Unsupported file type: {extension}"
    )



@api.get("/documents")
def get_documents() -> Dict[str, Any]:

    try:

        documents = list_documents()

        return {
            "documents": documents,
            "count": len(documents),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load documents: "
                f"{exc}"
            ),
        )


@api.post("/documents/sync")
def sync_documents() -> Dict[str, Any]:
    """Rebuild the local registry + keyword index from Pinecone."""

    try:
        synced = sync_registry_from_pinecone()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync documents: {exc}",
        )

    return {
        "message": f"Synced {synced} document(s) from Pinecone.",
        "documents": list_documents(),
    }


@api.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were provided.",
        )

    max_files = 5

    if len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {max_files} files are allowed "
                "per upload request."
            ),
        )

    uploaded: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    temp_paths: List[str] = []

    try:
        for uploaded_file in files:
            original_filename = uploaded_file.filename or ""

            if not original_filename:
                rejected.append(
                    {
                        "file_name": "",
                        "reason": "Missing filename.",
                    }
                )
                continue

            normalized_filename = original_filename.replace(
                "\\",
                "/",
            )
            filename = Path(normalized_filename).name

            if not filename or filename in {".", ".."}:
                rejected.append(
                    {
                        "file_name": original_filename,
                        "reason": "Invalid filename.",
                    }
                )
                continue

            extension = Path(filename).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": (
                            "Unsupported file type. Supported types: "
                            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
                        ),
                    }
                )
                continue

            contents = await uploaded_file.read(
                MAX_UPLOAD_SIZE_BYTES + 1
            )

            if len(contents) > MAX_UPLOAD_SIZE_BYTES:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": (
                            f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
                        ),
                    }
                )
                await uploaded_file.close()
                continue

            if not contents:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": "File is empty.",
                    }
                )
                await uploaded_file.close()
                continue

            valid, validation_reason = validate_uploaded_content(
                filename,
                contents,
            )

            if not valid:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": validation_reason,
                    }
                )
                await uploaded_file.close()
                continue

            temp_dir = tempfile.mkdtemp(
                prefix="agentic_rag_"
            )

            safe_filename = Path(filename).name
            temp_path = os.path.join(
                temp_dir,
                safe_filename,
            )

            with open(temp_path, "wb") as output_file:
                output_file.write(contents)

            temp_paths.append(temp_path)

            uploaded.append(
                {
                    "file_name": filename,
                    "temp_path": temp_path,
                    "size_bytes": len(contents),
                    "extension": extension,
                }
            )

            await uploaded_file.close()

        if not uploaded:
            return {
                "message": "No files were uploaded.",
                "uploaded": [],
                "rejected": rejected,
            }

        ingestion = await run_in_threadpool(
            build_vectorstore,
            [
                item["temp_path"]
                for item in uploaded
            ],
        )
        ingestion = ingestion or {}

        added = ingestion.get("added", [])
        message_parts = [f"Indexed {len(added)} file(s)."]
        if ingestion.get("skipped_existing"):
            message_parts.append(
                f"{len(ingestion['skipped_existing'])} already in the knowledge base."
            )
        if ingestion.get("no_text"):
            message_parts.append(
                "No extractable text (scanned PDF?): "
                + ", ".join(ingestion["no_text"])
            )
        if ingestion.get("failed"):
            message_parts.append(
                "Could not be parsed: "
                + ", ".join(ingestion["failed"])
            )

        return {
            "message": " ".join(message_parts),
            "ingestion": ingestion,
            "uploaded": [
                {
                    "file_name": item["file_name"],
                    "size_bytes": item["size_bytes"],
                    "extension": item["extension"],
                }
                for item in uploaded
            ],
            "rejected": rejected,
            "documents": await run_in_threadpool(list_documents),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"---UPLOAD ERROR: {exc}---")
        error_message = str(exc)
        raise HTTPException(
            status_code=500,
            detail=(
                "Upload failed while processing the documents. "
                f"{error_message}"
            ),
        ) from exc

    finally:
        for temp_path in temp_paths:
            try:
                temp_dir = os.path.dirname(temp_path)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if os.path.isdir(temp_dir):
                    shutil.rmtree(
                        temp_dir,
                        ignore_errors=True,
                    )

            except OSError:
                pass


@api.delete(
    "/documents/{document_id}"
)
def remove_document(
    document_id: str,
) -> Dict[str, Any]:

    try:

        deleted = delete_document(
            document_id
        )

        if not deleted:

            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return {
            "message": "Document deleted.",
            "document_id": document_id,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete document: "
                f"{exc}"
            ),
        )


# ============================================================================
# CHAT
# ============================================================================


@api.post("/chat")
async def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    observer = RunObserver(question)
    start_time = perf_counter()
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # The graph is synchronous (LLM + Pinecone calls); running it
        # directly in an async endpoint would block every other request.
        result = await run_in_threadpool(
            rag_app.invoke,
            {
                "question": question,
            },
            config=config,
        )

        latency = perf_counter() - start_time
        result = dict(result or {})

        # LangGraph returns a __interrupt__ entry when the workflow
        # pauses for human approval. This is a pending state, not a
        # completed answer.
        interrupts = result.get("__interrupt__") or []

        if interrupts:
            interrupt_value = getattr(
                interrupts[0],
                "value",
                interrupts[0],
            )

            if not isinstance(interrupt_value, dict):
                interrupt_value = {
                    "type": "web_search_approval",
                    "title": "Web search approval required",
                    "message": str(interrupt_value),
                }

            route = normalise_route(result)

            observer.record(
                route=route,
                retrieved_documents=safe_int(
                    result.get("retrieved_documents", 0)
                ),
                relevant_documents=safe_int(
                    result.get("relevant_documents", 0)
                ),
                grounded=False,
                answers_question=False,
                retry_count=safe_int(
                    result.get("retry_count", 0)
                ),
                latency_seconds=latency,
                sources=[],
                success=True,
                hitl_status="pending",
                hitl_reason=safe_str(
                    interrupt_value.get(
                        "message",
                        "Human approval is required before web search.",
                    )
                ),
            )

            return {
                "status": "approval_required",
                "thread_id": thread_id,
                "question": question,
                "answer": "",
                "generation": "",
                "route": "web",
                "sources": [],
                "retrieved_documents": safe_int(
                    result.get("retrieved_documents", 0)
                ),
                "relevant_documents": safe_int(
                    result.get("relevant_documents", 0)
                ),
                "grounded": False,
                "answers_question": False,
                "retry_count": safe_int(
                    result.get("retry_count", 0)
                ),
                "latency_seconds": round(latency, 4),
                "hitl_status": "pending",
                "hitl_reason": safe_str(
                    interrupt_value.get(
                        "message",
                        "Human approval is required before web search.",
                    )
                ),
                "hitl": interrupt_value,
                "security_status": result.get("security_status"),
                "security_reason": result.get("security_reason"),
                "security_redactions": result.get(
                    "security_redactions", []
                ) or [],
            }

        sources = result.get("sources", [])

        observer.record(
            route=str(
                result.get(
                    "route",
                    "unknown",
                )
            ),
            retrieved_documents=safe_int(
                result.get(
                    "retrieved_documents",
                    0,
                )
            ),
            relevant_documents=safe_int(
                result.get(
                    "relevant_documents",
                    0,
                )
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            answers_question=bool(
                result.get(
                    "answers_question",
                    False,
                )
            ),
            retry_count=int(
                result.get(
                    "retry_count",
                    0,
                )
                or 0
            ),
            latency_seconds=latency,
            sources=sources,
            success=True,
            hitl_status=str(
                result.get(
                    "hitl_status",
                    "",
                )
                or ""
            ),
            hitl_reason=str(
                result.get(
                    "hitl_reason",
                    "",
                )
                or ""
            ),
        )

        record_evaluation(
            question=question,
            result=result,
            latency_seconds=latency,
        )

        forget_thread(thread_id)

        return {
            "status": "completed",
            "thread_id": thread_id,
            "question": question,
            "answer": safe_str(
                result.get(
                    "answer",
                    result.get("generation", ""),
                )
            ),
            "generation": safe_str(
                result.get("generation", "")
            ),
            "route": normalise_route(result),
            "sources": sources or [],
            "retrieved_documents": safe_int(
                result.get("retrieved_documents", len(result.get("documents", []) or []))
            ),
            "relevant_documents": safe_int(
                result.get("relevant_documents", 0)
            ),
            "grounded": safe_bool(
                result.get("grounded", False)
            ),
            "answers_question": safe_bool(
                result.get("answers_question", False)
            ),
            "retry_count": safe_int(
                result.get("retry_count", 0)
            ),
            "latency_seconds": round(latency, 4),
            "hitl_status": result.get("hitl_status"),
            "hitl_reason": result.get("hitl_reason"),
            "security_status": result.get("security_status"),
            "security_reason": result.get("security_reason"),
            "security_redactions": result.get(
                "security_redactions", []
            ) or [],
        }

    except Exception as exc:
        latency = perf_counter() - start_time

        observer.record(
            route="error",
            latency_seconds=latency,
            success=False,
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================================
# HITL APPROVAL
# ============================================================================


@api.post("/chat/approval")
def chat_approval(
    request: HITLRequest,
) -> Dict[str, Any]:

    thread_id = request.thread_id

    if not thread_id:
        raise HTTPException(
            status_code=400,
            detail="thread_id is required.",
        )

    start_time = perf_counter()

    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # IMPORTANT: an interrupted LangGraph workflow must be resumed
        # with Command(resume=...), not with a normal state dictionary.
        result = rag_app.invoke(
            Command(resume=request.approved),
            config=config,
        )

        latency = perf_counter() - start_time
        result = dict(result or {})

        # If another interrupt is returned, surface it to the client
        # instead of pretending that an answer was completed.
        interrupts = result.get("__interrupt__") or []

        if interrupts:
            interrupt_value = getattr(
                interrupts[0],
                "value",
                interrupts[0],
            )

            if not isinstance(interrupt_value, dict):
                interrupt_value = {
                    "type": "web_search_approval",
                    "title": "Web search approval required",
                    "message": str(interrupt_value),
                }

            return {
                "status": "approval_required",
                "thread_id": thread_id,
                "question": safe_str(result.get("question", "")),
                "answer": "",
                "generation": "",
                "route": "web",
                "sources": [],
                "retrieved_documents": safe_int(
                    result.get("retrieved_documents", 0)
                ),
                "relevant_documents": safe_int(
                    result.get("relevant_documents", 0)
                ),
                "grounded": False,
                "answers_question": False,
                "retry_count": safe_int(
                    result.get("retry_count", 0)
                ),
                "latency_seconds": round(latency, 4),
                "hitl_status": "pending",
                "hitl_reason": safe_str(
                    interrupt_value.get(
                        "message",
                        "Human approval is required.",
                    )
                ),
                "hitl": interrupt_value,
                "security_status": result.get("security_status"),
                "security_reason": result.get("security_reason"),
                "security_redactions": result.get(
                    "security_redactions", []
                ) or [],
            }

        question = safe_str(
            result.get(
                "question",
                "",
            )
        )

        record_evaluation(
            question=question,
            result=result,
            latency_seconds=latency,
        )

        forget_thread(thread_id)

        return {
            "status": "completed",
            "thread_id": thread_id,
            "question": question,
            "answer": safe_str(
                result.get(
                    "answer",
                    result.get(
                        "generation",
                        "",
                    ),
                )
            ),
            "generation": safe_str(
                result.get(
                    "generation",
                    "",
                )
            ),
            "route": normalise_route(
                result
            ),
            "sources": result.get(
                "sources",
                [],
            ) or [],
            "retrieved_documents": safe_int(
                result.get(
                    "retrieved_documents",
                    len(
                        result.get(
                            "documents",
                            [],
                        ) or []
                    ),
                )
            ),
            "relevant_documents": safe_int(
                result.get(
                    "relevant_documents",
                    0,
                )
            ),
            "grounded": safe_bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            "answers_question": safe_bool(
                result.get(
                    "answers_question",
                    False,
                )
            ),
            "retry_count": safe_int(
                result.get(
                    "retry_count",
                    0,
                )
            ),
            "latency_seconds": round(
                latency,
                4,
            ),
            "hitl_status": result.get(
                "hitl_status"
            ),
            "hitl_reason": result.get(
                "hitl_reason"
            ),
            "security_status": result.get(
                "security_status"
            ),
            "security_reason": result.get(
                "security_reason"
            ),
            "security_redactions": (
                result.get(
                    "security_redactions",
                    [],
                ) or []
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================================
# EVALUATION API
# ============================================================================


@api.get("/evaluation")
def get_evaluation() -> Dict[str, Any]:

    try:

        tracker = EvaluationTracker()

        return {
            "results": [
                result.to_dict()
                for result in tracker.results
            ],
            "summary": tracker.summary(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load evaluation "
                f"history: {exc}"
            ),
        )


@api.delete("/evaluation")
def clear_evaluation() -> Dict[str, str]:

    try:

        tracker = EvaluationTracker()

        tracker.clear()

        return {
            "message": (
                "Evaluation history cleared."
            )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to clear evaluation "
                f"history: {exc}"
            ),
        )


# ============================================================================
# OBSERVABILITY API
# ============================================================================


@api.get("/observability")
def get_observability():
    records = load_observability_history()

    return {
        "runs": records,
        "summary": calculate_observability_summary(
            records
        ),
    }


@api.delete("/observability")
def clear_observability() -> Dict[str, str]:

    try:

        clear_observability_history()

        return {
            "message": (
                "Observability history cleared."
            )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to clear observability "
                f"history: {exc}"
            ),
        )


# ============================================================================
# DEVELOPMENT ENTRY POINT
# ============================================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:api",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )