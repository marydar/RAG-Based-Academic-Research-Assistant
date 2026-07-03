import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_service import (
    EMBEDDING_DIMENSION,
    EMBEDDINGS_PATH,
    EmbeddingService,
    load_chunks,
)


def main():
    chunks = load_chunks()

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "First run: python src\\embedding_service.py"
        )

    with np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    ) as data:
        embeddings = data["embeddings"]
        saved_chunk_ids = (
            data["chunk_ids"]
            .astype(str)
            .tolist()
        )

    expected_ids = [
        str(chunk["chunk_id"])
        for chunk in chunks
    ]

    expected_shape = (
        len(chunks),
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise AssertionError(
            f"Wrong shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    if saved_chunk_ids != expected_ids:
        raise AssertionError(
            "Chunk IDs are not stored correctly."
        )

    if not np.isfinite(embeddings).all():
        raise AssertionError(
            "Embeddings contain NaN or infinity."
        )

    document_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        document_norms,
        1.0,
        atol=0.001,
    ):
        raise AssertionError(
            "Document embeddings are not normalized."
        )

    service = EmbeddingService()

    question = (
        "مدل DETR روی چه مجموعه داده‌ای "
        "ارزیابی شده است؟"
    )

    query_embedding = service.encode_query(question)

    if query_embedding.shape != (
        EMBEDDING_DIMENSION,
    ):
        raise AssertionError(
            f"Wrong query shape: "
            f"{query_embedding.shape}"
        )

    query_norm = np.linalg.norm(
        query_embedding
    )

    if not np.isclose(
        query_norm,
        1.0,
        atol=0.001,
    ):
        raise AssertionError(
            "Query embedding is not normalized."
        )

    print(f"[PASS] Chunk count: {len(chunks)}")
    print(f"[PASS] Embeddings shape: {embeddings.shape}")
    print(f"[PASS] Query shape: {query_embedding.shape}")
    print(f"[PASS] Query norm: {query_norm:.6f}")
    print("ALL EMBEDDING TESTS PASSED")


if __name__ == "__main__":
    main()