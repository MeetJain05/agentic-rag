"""
Checkpointer for LangGraph state (needed to resume a run after the
human-in-the-loop web-search approval).

Selection order:
1. CHECKPOINT_DATABASE_URL (Postgres) - survives restarts and
   redeploys. Required for durable approvals on hosts with an
   ephemeral disk (e.g. Render free tier). Neon/Supabase free tiers work.
2. SQLite file in DATA_DIR - survives process restarts on the same
   disk (local dev, Docker with a mounted volume).
3. In-memory - last resort if neither is available.
"""

import os
import sqlite3
from pathlib import Path

from backend.config import DATA_DIR


def _postgres_saver(url: str):
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # Small pool: free-tier Postgres plans cap connections, and the
    # app handles few concurrent requests on a small instance.
    pool = ConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=4,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def _sqlite_saver():
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(DATA_DIR) / "checkpoints.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def create_checkpointer():
    url = os.getenv("CHECKPOINT_DATABASE_URL", "").strip()

    if url:
        try:
            saver = _postgres_saver(url)
            print("---CHECKPOINTER: POSTGRES---")
            return saver
        except Exception as exc:
            print(f"---POSTGRES CHECKPOINTER UNAVAILABLE: {exc}---")

    try:
        saver = _sqlite_saver()
        print("---CHECKPOINTER: SQLITE---")
        return saver
    except Exception as exc:
        print(f"---SQLITE CHECKPOINTER UNAVAILABLE: {exc}---")

    from langgraph.checkpoint.memory import InMemorySaver

    print("---CHECKPOINTER: IN-MEMORY (approvals lost on restart)---")
    return InMemorySaver()
