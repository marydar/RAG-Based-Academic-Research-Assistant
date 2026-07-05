"""Store chunks and embeddings in ChromaDB."""

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

def get_chunk_id(chunk):
    """Return the string ID of a chunk."""

    chunk_id = chunk.get("chunk_id")

    if not chunk_id:
        raise ValueError("A chunk has no chunk_id.")

    return str(chunk_id)

def load_chunks():
    """Load all chunks."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        chunks = data.get("chunks", [])
    else:
        chunks = data

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "No chunks were found."
        )

    return chunks


def load_embeddings():
    """Load embeddings, IDs and embedding information."""

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

        embedding_ids = (
            data["chunk_ids"]
            .astype(str)
            .tolist()
        )

    with INFO_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        embedding_info = json.load(file)

    return (
        embeddings,
        embedding_ids,
        embedding_info,
    )


def create_metadata(chunk):
    """Create ChromaDB metadata for one chunk."""

    page_start = int(
        chunk.get("page_start", 0)
    )

    page_end = int(
        chunk.get(
            "page_end",
            page_start,
        )
    )

    section = chunk.get(
        "section",
        chunk.get(
            "section_name",
            "unknown",
        ),
    )

    retrieval_enabled = bool(
        chunk.get(
            "retrieval_enabled",
            True,
        )
    )

    if section == "references":
        retrieval_enabled = False

    return {
        "paper_id": str(
            chunk.get("paper_id", "unknown")
        ),
        "paper_title": str(
            chunk.get("paper_title", "unknown")
        ),
        "paper_short_name": str(
            chunk.get(
                "paper_short_name",
                "unknown",
            )
        ),
        "section": str(section),
        "section_title": str(
            chunk.get(
                "section_title",
                section,
            )
        ),
        "chunk_type": str(
            chunk.get(
                "chunk_type",
                "text",
            )
        ),
        "page_start": page_start,
        "page_end": page_end,
        "retrieval_enabled": retrieval_enabled,
    }


def build_vector_store():
    """Rebuild the ChromaDB collection."""

    chunks = load_chunks()

    (
        embeddings,
        embedding_ids,
        embedding_info,
    ) = load_embeddings()

    chunk_ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        document = chunk.get("text")

        if not chunk_id:
            raise ValueError(
                "A chunk has no chunk_id."
            )

        if (
            not isinstance(document, str)
            or not document.strip()
        ):
            raise ValueError(
                f"Chunk {chunk_id} has no text."
            )

        chunk_ids.append(str(chunk_id))
        documents.append(document.strip())
        metadatas.append(
            create_metadata(chunk)
        )

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs were found."
        )

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
            f"Wrong embedding shape: "
            f"{embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or Inf."
        )

    DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Old collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
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

    stored_count = collection.count()

    if stored_count != len(chunks):
        raise ValueError(
            f"Stored records: {stored_count}. "
            f"Expected: {len(chunks)}."
        )

    print()
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Expected records: {len(chunks)}")
    print(f"Stored records: {stored_count}")
    print("Distance type: cosine")
    print(f"Database path: {DB_PATH}")
    print("Vector store created successfully.")


if __name__ == "__main__":
    build_vector_store()
