import json
from pathlib import Path

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "all_chunks.json"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "all_embeddings.npz"
)

INFO_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "embedding_info.json"
)

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "chroma_db"
)

COLLECTION_NAME = "academic_research_chunks"


def load_chunks():
    """Chunkها را از فایل JSON می‌خواند."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        chunks = data.get("chunks", [])
    else:
        chunks = data

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No chunks were found.")

    return chunks


def load_embedding_data():
    """Embeddingها، شناسه‌ها و اطلاعات مدل را می‌خواند."""

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    if not INFO_PATH.exists():
        raise FileNotFoundError(
            f"Embedding information not found: {INFO_PATH}"
        )

    with np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    ) as data:
        embeddings = data["embeddings"].astype(
            np.float32
        )

        chunk_ids = (
            data["chunk_ids"]
            .astype(str)
            .tolist()
        )

    with open(
        INFO_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        embedding_info = json.load(file)

    return embeddings, chunk_ids, embedding_info


def get_value(chunk, *keys, default=""):
    """یک مقدار را از Chunk یا metadata آن پیدا می‌کند."""

    metadata = chunk.get("metadata", {})

    for key in keys:
        value = chunk.get(key)

        if value not in (None, ""):
            return value

        if isinstance(metadata, dict):
            value = metadata.get(key)

            if value not in (None, ""):
                return value

    return default


def get_chunk_id(chunk):
    """شناسه Chunk را برمی‌گرداند."""

    chunk_id = get_value(
        chunk,
        "chunk_id",
        "id",
    )

    if not chunk_id:
        raise ValueError(
            "A chunk has no chunk_id."
        )

    return str(chunk_id)


def get_document(chunk):
    """متن اصلی Chunk را برای ذخیره در Chroma آماده می‌کند."""

    document = get_value(
        chunk,
        "text",
        "chunk_text",
        "content",
        "embedding_text",
    )

    if not isinstance(document, str):
        raise ValueError(
            f"Chunk {get_chunk_id(chunk)} has no text."
        )

    document = document.strip()

    if not document:
        raise ValueError(
            f"Chunk {get_chunk_id(chunk)} has empty text."
        )

    return document


def get_metadata(chunk):
    """Metadata موردنیاز Retrieval را می‌سازد."""

    paper_id = get_value(
        chunk,
        "paper_id",
        default="unknown",
    )

    paper_title = get_value(
        chunk,
        "paper_title",
        "paper_name",
        default=str(paper_id),
    )

    section = get_value(
        chunk,
        "section",
        "section_name",
        "section_title",
        default="unknown",
    )

    page_start = get_value(
        chunk,
        "page_start",
        "start_page",
        default="unknown",
    )

    page_end = get_value(
        chunk,
        "page_end",
        "end_page",
        default=page_start,
    )

    retrieval_enabled = get_value(
        chunk,
        "retrieval_enabled",
        default=True,
    )

    if isinstance(retrieval_enabled, str):
        retrieval_enabled = (
            retrieval_enabled.lower()
            not in {"false", "0", "no"}
        )
    else:
        retrieval_enabled = bool(
            retrieval_enabled
        )

    # بخش References در جست‌وجوی عادی غیرفعال است.
    if str(section).strip().lower() == "references":
        retrieval_enabled = False

    return {
        "paper_id": str(paper_id),
        "paper_title": str(paper_title),
        "section": str(section),
        "page_start": str(page_start),
        "page_end": str(page_end),
        "retrieval_enabled": retrieval_enabled,
    }


def build_vector_store():
    """Chunkها و Embeddingها را در ChromaDB ذخیره می‌کند."""

    chunks = load_chunks()

    (
        embeddings,
        embedding_ids,
        embedding_info,
    ) = load_embedding_data()

    chunk_ids = [
        get_chunk_id(chunk)
        for chunk in chunks
    ]

    if chunk_ids != embedding_ids:
        raise ValueError(
            "Chunk IDs do not match embedding IDs."
        )

    expected_dimension = int(
        embedding_info["embedding_dimension"]
    )

    expected_shape = (
        len(chunks),
        expected_dimension,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Wrong embedding shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    documents = [
        get_document(chunk)
        for chunk in chunks
    ]

    metadatas = [
        get_metadata(chunk)
        for chunk in chunks
    ]

    DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored records: {collection.count()}")
    print(f"Database path: {DB_PATH}")

    return collection


if __name__ == "__main__":
    build_vector_store()