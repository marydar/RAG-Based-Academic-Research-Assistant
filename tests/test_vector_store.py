import json
import sys
from pathlib import Path

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import (
    COLLECTION_NAME,
    DB_PATH,
    INFO_PATH,
    build_vector_store,
    get_chunk_id,
    load_chunks,
)


def main():
    chunks = load_chunks()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "First run: python src\\vector_store.py"
        )

    with open(
        INFO_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        embedding_info = json.load(file)

    expected_dimension = int(
        embedding_info["embedding_dimension"]
    )

    expected_count = len(chunks)

    # اتصال دوباره، برای تست ماندگاری اطلاعات
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    if collection.count() != expected_count:
        raise AssertionError(
            f"Wrong record count: {collection.count()}. "
            f"Expected: {expected_count}"
        )

    first_id = get_chunk_id(chunks[0])
    last_id = get_chunk_id(chunks[-1])

    result = collection.get(
        ids=[first_id, last_id],
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ],
    )

    if set(result["ids"]) != {
        first_id,
        last_id,
    }:
        raise AssertionError(
            "Sample chunk IDs were not found."
        )

    documents = result["documents"]
    metadatas = result["metadatas"]

    if not documents or any(
        not document
        for document in documents
    ):
        raise AssertionError(
            "Stored documents are empty."
        )

    required_metadata = {
        "paper_id",
        "paper_title",
        "section",
        "page_start",
        "page_end",
        "retrieval_enabled",
    }

    for metadata in metadatas:
        missing_fields = (
            required_metadata
            - set(metadata.keys())
        )

        if missing_fields:
            raise AssertionError(
                f"Missing metadata fields: "
                f"{missing_fields}"
            )

    sample_embeddings = np.asarray(
        result["embeddings"],
        dtype=np.float32,
    )

    expected_shape = (
        2,
        expected_dimension,
    )

    if sample_embeddings.shape != expected_shape:
        raise AssertionError(
            f"Wrong sample embedding shape: "
            f"{sample_embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    # اجرای دوباره Upsert برای تست جلوگیری از تکرار
    count_before = collection.count()

    build_vector_store()

    client_again = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection_again = client_again.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    count_after = collection_again.count()

    if count_after != count_before:
        raise AssertionError(
            "Duplicate records were created."
        )

    print(f"[PASS] Collection: {COLLECTION_NAME}")
    print(f"[PASS] Record count: {count_after}")
    print(
        f"[PASS] Sample embedding shape: "
        f"{sample_embeddings.shape}"
    )
    print("[PASS] Documents are stored")
    print("[PASS] Metadata is stored")
    print("[PASS] Database persistence works")
    print("[PASS] Duplicate prevention works")
    print("ALL VECTOR STORE TESTS PASSED")


if __name__ == "__main__":
    main()