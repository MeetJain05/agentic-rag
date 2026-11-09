"""
End-to-end benchmark for the RAG graph.

Runs every question in benchmark_dataset.json through the real graph
(Groq, Pinecone, Tavily) and reports routing accuracy, answer
correctness, grounding, answer quality, retrieval precision, latency
and retries.

It is isolated from the app's data: documents from backend/documents
are indexed into a separate Pinecone namespace ("benchmark" by
default) with its own local SQLite store, and that namespace is
deleted afterwards unless --keep is passed. Web-search approvals are
auto-approved.

    python backend/evaluation/run_benchmark.py
    python backend/evaluation/run_benchmark.py --keep --delay 5

Writes benchmark_results.json and benchmark_results.md next to this
file.
"""

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
DATASET_FILE = EVAL_DIR / "benchmark_dataset.json"
RESULTS_JSON = EVAL_DIR / "benchmark_results.json"
RESULTS_MD = EVAL_DIR / "benchmark_results.md"
KNOWLEDGE_DIR = PROJECT_ROOT / "backend" / "documents"

RATE_LIMIT_WAIT_SECONDS = 45
MAX_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--namespace", default="benchmark")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds between questions (Groq rate limits)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the benchmark namespace afterwards")
    return parser.parse_args()


ARGS = parse_args()

# Configuration is read at import time, so isolate before importing
# anything from the backend.
os.environ["PINECONE_NAMESPACE"] = ARGS.namespace
os.environ["DATA_DIR"] = str(PROJECT_ROOT / "data" / f"eval-{ARGS.namespace}")
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)
os.environ["PINECONE_NAMESPACE"] = ARGS.namespace

from langgraph.types import Command  # noqa: E402

from backend.config import PINECONE_NAMESPACE  # noqa: E402
from backend.graph.graph import app, checkpointer  # noqa: E402
from backend.ingestion import build_vectorstore  # noqa: E402
from backend.retrieval import get_pinecone_index  # noqa: E402

assert PINECONE_NAMESPACE == ARGS.namespace, "namespace isolation failed"


# ---------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------


def namespace_vector_count() -> int:
    stats = get_pinecone_index().describe_index_stats()
    namespaces = stats.get("namespaces", {}) or {}
    return int((namespaces.get(ARGS.namespace) or {}).get("vector_count", 0))


def index_knowledge_base() -> None:
    files = sorted(str(p) for p in KNOWLEDGE_DIR.iterdir() if p.is_file())
    print(f"Indexing {len(files)} files into namespace '{ARGS.namespace}'...")
    summary = build_vectorstore(files)
    print(f"  added={len(summary['added'])} existing="
          f"{len(summary['skipped_existing'])} failed={len(summary['failed'])}")

    # Pinecone is eventually consistent; wait until vectors are visible.
    for _ in range(30):
        if namespace_vector_count() > 0:
            break
        time.sleep(2)
    time.sleep(5)
    print(f"  {namespace_vector_count()} chunks searchable")


def delete_namespace() -> None:
    try:
        get_pinecone_index().delete(delete_all=True, namespace=ARGS.namespace)
        print(f"Deleted namespace '{ARGS.namespace}'.")
    except Exception as exc:
        print(f"Could not delete namespace '{ARGS.namespace}': {exc}")


# ---------------------------------------------------------------------
# Running questions
# ---------------------------------------------------------------------


def is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(s in text for s in ("rate limit", "rate_limit", "429",
                                   "tokens per minute"))


def run_graph(question: str) -> Dict[str, Any]:
    thread_id = f"benchmark-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()
    try:
        result = dict(app.invoke({"question": question}, config=config))
        approvals = 0
        # Auto-approve web search so web questions can be scored.
        while result.get("__interrupt__") and approvals < 3:
            approvals += 1
            result = dict(app.invoke(Command(resume=True), config=config))
        result["latency_seconds"] = time.perf_counter() - start
        return result
    finally:
        try:
            checkpointer.delete_thread(thread_id)
        except Exception:
            pass


def contains_expected(answer: str, expected: List[str]) -> bool:
    """Every term must appear; 'a|b' means either a or b."""
    lowered = answer.lower()
    return all(
        any(option.lower() in lowered for option in term.split("|"))
        for term in expected
    )


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    expected_route = item["expected_route"]
    expected_terms = item.get("expected_answer_contains", [])

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = run_graph(question)
            break
        except Exception as exc:
            last_error = exc
            if is_rate_limit_error(exc) and attempt < MAX_ATTEMPTS - 1:
                print(f"    rate limited, waiting {RATE_LIMIT_WAIT_SECONDS}s")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            return {
                "question": question,
                "category": item.get("category", ""),
                "expected_route": expected_route,
                "actual_route": "error",
                "error": str(exc)[:300],
                "rate_limited": is_rate_limit_error(exc),
                "passed": False,
            }
    else:  # pragma: no cover
        raise RuntimeError(str(last_error))

    answer = str(result.get("answer") or result.get("generation") or "")
    route = str(result.get("route", "unknown"))
    if route == "web_rejected":
        route = "web"

    routing_correct = route == expected_route
    answer_correct = contains_expected(answer, expected_terms)
    grounded = bool(result.get("grounded", False))
    answers_question = bool(result.get("answers_question", False))

    return {
        "question": question,
        "category": item.get("category", ""),
        "expected_route": expected_route,
        "actual_route": route,
        "routing_correct": routing_correct,
        "answer_contains_expected": answer_correct,
        "grounded": grounded,
        "answers_question": answers_question,
        "retrieved_documents": int(result.get("retrieved_documents", 0) or 0),
        "relevant_documents": int(result.get("relevant_documents", 0) or 0),
        "retry_count": int(result.get("retry_count", 0) or 0),
        "latency_seconds": round(float(result["latency_seconds"]), 2),
        "answer": answer,
        "passed": routing_correct and answer_correct and grounded and answers_question,
        "error": None,
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def rate(values: List[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct * (len(ordered) - 1))))
    return round(ordered[index], 2)


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    rag = [r for r in ok if r["expected_route"] != "greeting"]
    local = [r for r in ok if r["actual_route"] == "local"]

    precision = [
        r["relevant_documents"] / r["retrieved_documents"]
        for r in local if r["retrieved_documents"]
    ]

    def latency_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        values = [r["latency_seconds"] for r in rows]
        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 2) if values else 0.0,
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({r.get("category", "") for r in results}):
        rows = [r for r in results if r.get("category") == category]
        by_category[category] = {
            "questions": len(rows),
            "pass_rate": rate([r["passed"] for r in rows]),
        }

    return {
        "questions": len(results),
        "errors": len(results) - len(ok),
        "rate_limited": sum(bool(r.get("rate_limited")) for r in results),
        "routing_accuracy": rate([r["routing_correct"] for r in ok]),
        "answer_correctness": rate([r["answer_contains_expected"] for r in rag]),
        "grounded_rate": rate([r["grounded"] for r in rag]),
        "answer_quality_rate": rate([r["answers_question"] for r in rag]),
        "retrieval_precision": round(statistics.mean(precision), 4) if precision else 0.0,
        "average_retries": round(statistics.mean([r["retry_count"] for r in ok]), 2) if ok else 0.0,
        "pass_rate": rate([r["passed"] for r in results]),
        "latency_seconds": {
            "all": latency_stats(ok),
            "local": latency_stats([r for r in ok if r["actual_route"] == "local"]),
            "web": latency_stats([r for r in ok if r["actual_route"] == "web"]),
            "greeting": latency_stats([r for r in ok if r["actual_route"] == "greeting"]),
        },
        "by_category": by_category,
    }


def write_markdown(summary: Dict[str, Any], results: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    pct = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    lat = summary["latency_seconds"]
    lines = [
        "# Benchmark results",
        "",
        f"Run on {meta['date']} with `{meta['llm_model']}` (Groq), "
        f"{summary['questions']} questions. Generated by "
        "`python backend/evaluation/run_benchmark.py`.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Routing accuracy | {pct(summary['routing_accuracy'])} |",
        f"| Answer contains expected facts | {pct(summary['answer_correctness'])} |",
        f"| Grounded (hallucination check passed) | {pct(summary['grounded_rate'])} |",
        f"| Answers the question (answer grader) | {pct(summary['answer_quality_rate'])} |",
        f"| Retrieval precision (relevant / retrieved) | {pct(summary['retrieval_precision'])} |",
        f"| Average retries | {summary['average_retries']} |",
        f"| End-to-end pass rate | {pct(summary['pass_rate'])} |",
        "",
        "| Route | Questions | Median latency | p95 latency |",
        "|---|---|---|---|",
    ]
    for route in ("local", "web", "greeting"):
        s = lat[route]
        if s["count"]:
            lines.append(f"| {route} | {s['count']} | {s['p50']}s | {s['p95']}s |")
    lines += [
        "",
        "| Question | Expected | Got | Facts | Grounded | Pass |",
        "|---|---|---|---|---|---|",
    ]
    tick = lambda v: "yes" if v else "no"  # noqa: E731
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['question']} | {r['expected_route']} | error | | | no |")
            continue
        lines.append(
            f"| {r['question']} | {r['expected_route']} | {r['actual_route']} | "
            f"{tick(r['answer_contains_expected'])} | {tick(r['grounded'])} | {tick(r['passed'])} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))

    index_knowledge_base()

    results: List[Dict[str, Any]] = []
    try:
        for number, item in enumerate(dataset, start=1):
            print(f"[{number}/{len(dataset)}] {item['question']}")
            outcome = evaluate(item)
            results.append(outcome)
            status = "PASS" if outcome["passed"] else "FAIL"
            print(f"    {status} route={outcome.get('actual_route')} "
                  f"latency={outcome.get('latency_seconds', '-')}s "
                  f"{outcome.get('error') or ''}")
            if number < len(dataset):
                time.sleep(ARGS.delay)
    finally:
        if not ARGS.keep:
            delete_namespace()

    summary = summarise(results)
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "llm_model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        "namespace": ARGS.namespace,
    }
    RESULTS_JSON.write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, results, meta)

    print()
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {RESULTS_JSON.name} and {RESULTS_MD.name}")


if __name__ == "__main__":
    main()
