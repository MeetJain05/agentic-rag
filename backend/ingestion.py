import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from backend import store
from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_LAYOUT_MODEL,
    PDF_OCR,
    PDF_OCR_LANGUAGE,
    PDF_PARSER,
    PDF_TABLES,
    PINECONE_NAMESPACE,
    SUPPORTED_EXTENSIONS,
)
from backend.retrieval import (
    get_pinecone_index,
    iter_records,
    list_ids,
    upsert_documents,
)


def calculate_document_id(file_path: str) -> str:
    """
    Generate a deterministic document ID from file contents.

    Identical files receive the same document ID even if their
    filenames are different.
    """

    path = Path(file_path)

    file_hash = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def _ocr_empty_pages(path: Path, pages: List[Dict[str, Any]]) -> None:
    """
    OCR only the pages that have no text layer (scanned pages), one
    page at a time to keep memory flat. No-op without Tesseract.
    """

    import pymupdf

    try:
        pymupdf.get_tessdata()
    except Exception:
        print("---TESSERACT NOT FOUND, SKIPPING OCR---")
        return

    with pymupdf.open(str(path)) as pdf:
        for position, page in enumerate(pages):
            if str(page.get("text", "")).strip():
                continue
            try:
                textpage = pdf[position].get_textpage_ocr(
                    language=PDF_OCR_LANGUAGE,
                    dpi=200,
                    full=True,
                )
                page["text"] = pdf[position].get_text(textpage=textpage)
                print(f"---OCR PAGE {position + 1}: {path.name}---")
            except Exception as exc:
                print(f"---OCR FAILED ON PAGE {position + 1}: {exc}---")


def _load_pdf_pymupdf(path: Path) -> List[Document]:
    """
    Layout-aware PDF extraction with PyMuPDF4LLM.

    Produces Markdown per page, so headings, lists and tables survive
    (pypdf flattens them into one run of text). Pages without a text
    layer are OCR'd with Tesseract when it is installed.
    """

    import pymupdf4llm

    # The layout model costs ~250 MB extra; keep it off on small hosts.
    pymupdf4llm.use_layout(PDF_LAYOUT_MODEL)

    options: Dict[str, Any] = {
        "page_chunks": True,
        "show_progress": False,
    }
    if PDF_LAYOUT_MODEL:
        options.update(use_ocr=PDF_OCR, ocr_language=PDF_OCR_LANGUAGE)
    elif not PDF_TABLES:
        # Table detection is ~85% of parse time; opt-in via PDF_TABLES.
        options["table_strategy"] = None

    pages = pymupdf4llm.to_markdown(str(path), **options)

    if PDF_OCR and not PDF_LAYOUT_MODEL:
        _ocr_empty_pages(path, pages)

    documents = []
    for position, page in enumerate(pages):
        meta = page.get("metadata", {}) or {}
        page_number = int(meta.get("page_number") or position + 1)
        documents.append(
            Document(
                page_content=str(page.get("text", "")),
                metadata={
                    # Same convention as PyPDFLoader: 0-based page,
                    # 1-based human-readable label.
                    "page": page_number - 1,
                    "page_label": str(page_number),
                    "total_pages": int(meta.get("page_count") or len(pages)),
                    "title": str(meta.get("title") or ""),
                    "parser": "pymupdf4llm",
                },
            )
        )
    return documents


def _load_pdf(path: Path) -> List[Document]:
    if PDF_PARSER == "pymupdf":
        try:
            return _load_pdf_pymupdf(path)
        except ImportError:
            print("---PYMUPDF4LLM NOT INSTALLED, USING PYPDF---")
        except Exception as exc:
            print(f"---PYMUPDF FAILED ({exc}), USING PYPDF---")

    documents = PyPDFLoader(str(path)).load()
    for document in documents:
        document.metadata["parser"] = "pypdf"
    return documents


def load_document(file_path: str) -> List[Document]:
    """Load a supported document."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        documents = _load_pdf(path)

    elif extension == ".docx":
        documents = Docx2txtLoader(str(path)).load()

    elif extension in {".txt", ".md"}:
        documents = TextLoader(
            str(path),
            encoding="utf-8",
        ).load()

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    document_id = calculate_document_id(file_path)

    for document in documents:
        document.metadata.update(
            {
                # Store the file name, not the (temporary) server path.
                "source": path.name,
                "file_name": path.name,
                "document_id": document_id,
                "file_type": extension,
            }
        )

    return documents


def split_documents(
    documents: List[Document],
) -> List[Document]:
    """Split documents into smaller chunks."""

    # Markdown-aware separators: split on headings first, then
    # paragraphs, then sentences. Also fine for plain text.
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def document_exists(
    document_id: str,
) -> bool:
    """Check whether a document already exists."""

    if store.has_document(document_id):
        return True
    # Pinecone is the source of truth (registry may be fresh/empty).
    return bool(list_ids(prefix=document_id))


def _has_text(documents: List[Document]) -> bool:
    return any(document.page_content.strip() for document in documents)


def build_vectorstore(
    file_paths: List[str],
) -> Dict[str, List[str]]:
    """
    Load, split and add new documents to Pinecone.

    Existing documents are skipped. Returns a per-file summary so
    callers can report skipped/empty files instead of claiming
    success for everything.
    """

    summary: Dict[str, List[str]] = {
        "added": [],
        "skipped_existing": [],
        "no_text": [],
        "failed": [],
    }

    all_chunks: List[Document] = []
    chunk_ids: List[str] = []
    registrations = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            print(f"---FILE NOT FOUND: {file_path}---")
            summary["failed"].append(path.name)
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"---UNSUPPORTED FILE TYPE: {path.name}---")
            summary["failed"].append(path.name)
            continue

        document_id = calculate_document_id(file_path)

        if document_exists(document_id):
            print(f"---DOCUMENT ALREADY EXISTS: {path.name}---")
            summary["skipped_existing"].append(path.name)
            continue

        print(f"---LOADING: {path.name}---")

        try:
            documents = load_document(file_path)
        except Exception as exc:  # corrupt / encrypted / unparsable file
            print(f"---FAILED TO PARSE {path.name}: {exc}---")
            summary["failed"].append(path.name)
            continue

        # Scanned/image-only PDFs yield pages with empty text.
        if not _has_text(documents):
            print(f"---NO EXTRACTABLE TEXT: {path.name}---")
            summary["no_text"].append(path.name)
            continue

        chunks = [
            chunk
            for chunk in split_documents(documents)
            if chunk.page_content.strip()
        ]

        # Chunk index is per document, and the ID starts with the
        # document hash so prefix list/delete stays cheap.
        document_chunk_ids = []
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
            document_chunk_ids.append(f"{document_id}#{chunk_index:05d}")

        all_chunks.extend(chunks)
        chunk_ids.extend(document_chunk_ids)
        registrations.append(
            (document_id, path.name, path.suffix.lower(), chunks, document_chunk_ids)
        )
        summary["added"].append(path.name)

    if not all_chunks:
        if summary["no_text"] and not summary["skipped_existing"]:
            raise ValueError(
                "No readable text was found in "
                f"{', '.join(summary['no_text'])}. Upload a text-based "
                "PDF or an OCR-processed document."
            )
        print("---NO NEW DOCUMENTS TO ADD---")
        return summary

    upsert_documents(
        documents=all_chunks,
        ids=chunk_ids,
    )

    # Register only after Pinecone accepted the vectors, so the
    # registry never lists a document that isn't searchable.
    for document_id, file_name, file_type, chunks, ids in registrations:
        store.add_document(
            document_id=document_id,
            file_name=file_name,
            file_type=file_type,
            chunks=chunks,
            chunk_ids=ids,
        )

    print("---VECTOR STORE UPDATED---")
    print(f"Chunks added: {len(all_chunks)}")

    return summary


def list_documents() -> List[Dict[str, Any]]:
    """
    Return documents in the knowledge base from the local registry
    (one SQLite query instead of fetching every vector).
    """

    return store.list_documents()


def sync_registry_from_pinecone() -> int:
    """Rebuild the local registry + keyword index from Pinecone."""

    return store.sync_from_pinecone(iter_records())


def list_documents_from_pinecone() -> List[Dict[str, Any]]:
    """
    Slow path: scan every vector in Pinecone. Kept for diagnostics.
    """

    documents = {}
    for record in iter_records():
        metadata = record["metadata"]
        document_id = metadata.get("document_id")
        if document_id and document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "file_name": metadata.get("file_name", "Unknown"),
                "file_type": metadata.get("file_type", "Unknown"),
                "source": metadata.get("source", "Unknown"),
            }

    return list(documents.values())


def delete_document(
    document_id: str,
) -> bool:
    """
    Delete all chunks belonging to a document.

    Returns True when a document was deleted.
    """

    # Works for both the current "<hash>#<n>" IDs and the older
    # "<hash>-<file>-<n>-<chunkhash>" IDs, since both start with the hash.
    ids = list_ids(prefix=document_id)

    if not ids:
        # Clean up a stale registry entry, if any.
        was_registered = store.has_document(document_id)
        store.remove_document(document_id)
        return was_registered

    index = get_pinecone_index()
    for start in range(0, len(ids), 1000):
        index.delete(
            ids=ids[start:start + 1000],
            namespace=PINECONE_NAMESPACE,
        )

    store.remove_document(document_id)
    return True


def get_files_from_directory(
    directory: str,
) -> List[str]:
    """Return supported files from a directory."""

    path = Path(directory)

    if not path.exists():
        return []

    return [
        str(file)
        for file in path.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]