import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PINECONE_NAMESPACE,
    SUPPORTED_EXTENSIONS,
)
from backend.retrieval import (
    get_pinecone_index,
    iter_records,
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


def load_document(file_path: str) -> List[Document]:
    """Load a supported document."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension in {".txt", ".md"}:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    document_id = calculate_document_id(file_path)

    for document in documents:
        document.metadata.update(
            {
                "source": str(path),
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def document_exists(
    document_id: str,
) -> bool:
    """Check whether a document already exists."""

    return any(
        record["metadata"].get("document_id") == document_id
        for record in iter_records()
    )


def build_vectorstore(
    file_paths: List[str],
) -> None:
    """
    Load, split and add new documents to Pinecone.

    Existing documents are skipped.
    """

    all_documents = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            print(f"---FILE NOT FOUND: {file_path}---")
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"---UNSUPPORTED FILE TYPE: {path.name}---"
            )
            continue

        document_id = calculate_document_id(file_path)

        if document_exists(document_id):
            print(
                f"---DOCUMENT ALREADY EXISTS: {path.name}---"
            )
            continue

        print(f"---LOADING: {path.name}---")

        documents = load_document(file_path)
        all_documents.extend(documents)

    if not all_documents:
        print("---NO NEW DOCUMENTS TO ADD---")
        return

    document_chunks = split_documents(all_documents)

    if not document_chunks:
        file_names = ", ".join(
            sorted(
                {
                    document.metadata.get("file_name", "document")
                    for document in all_documents
                }
            )
        )
        raise ValueError(
            "No readable text was found in "
            f"{file_names}. Upload a text-based PDF or an OCR-processed "
            "document."
        )

    chunk_ids = []

    for index, document in enumerate(document_chunks):
        document_id = document.metadata["document_id"]

        chunk_content = document.page_content

        chunk_hash = hashlib.sha256(
            chunk_content.encode("utf-8")
        ).hexdigest()

        chunk_id = (
            f"{document_id}-"
            f"{document.metadata.get('file_name', 'document')}-"
            f"{index}-"
            f"{chunk_hash}"
        )

        document.metadata["chunk_index"] = index

        chunk_ids.append(chunk_id)

    upsert_documents(
        documents=document_chunks,
        ids=chunk_ids,
    )

    print("---VECTOR STORE UPDATED---")
    print(
        f"Chunks added: {len(document_chunks)}"
    )


def list_documents() -> List[Dict[str, Any]]:
    """
    Return unique documents currently stored
    in the knowledge base.
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

    ids = []

    for record in iter_records():
        if record["metadata"].get("document_id") == document_id:
            ids.append(record["id"])

    if not ids:
        return False

    get_pinecone_index().delete(
        ids=ids,
        namespace=PINECONE_NAMESPACE,
    )

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