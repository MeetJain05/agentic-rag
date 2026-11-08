"""Index local documents into the configured Pinecone namespace."""

import argparse
from pathlib import Path

from backend.ingestion import build_vectorstore, get_files_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index documents into Pinecone for Agentic Adaptive RAG."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories containing PDF, DOCX, TXT, or Markdown files.",
    )
    args = parser.parse_args()

    files: list[str] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(get_files_from_directory(str(path)))
        elif path.is_file():
            files.append(str(path))
        else:
            parser.error(f"Path does not exist: {raw_path}")

    if not files:
        parser.error("No supported documents were found.")

    build_vectorstore(files)


if __name__ == "__main__":
    main()
