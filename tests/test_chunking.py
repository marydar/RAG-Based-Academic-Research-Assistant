import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import SemanticRetriever


question_1 = "What AP does DETR achieve on COCO?"
question_2 = "What is the capital city of Japan?"


retriever = SemanticRetriever()

embedding_1 = (
    retriever.embedding_service.encode_query(
        question_1
    )
)

embedding_2 = (
    retriever.embedding_service.encode_query(
        question_2
    )
)

cosine_similarity = float(
    np.dot(
        embedding_1,
        embedding_2,
    )
)

maximum_difference = float(
    np.max(
        np.abs(
            embedding_1 - embedding_2
        )
    )
)

print()
print("=" * 70)
print("QUERY EMBEDDING CHECK")
print("=" * 70)

print(
    f"Cosine similarity between questions: "
    f"{cosine_similarity:.8f}"
)

print(
    f"Maximum vector difference: "
    f"{maximum_difference:.8f}"
)

print(
    "Embeddings are identical:",
    np.allclose(
        embedding_1,
        embedding_2,
        atol=1e-7,
    ),
)


results_1 = retriever.search(
    question_1,
    top_k=3,
)

results_2 = retriever.search(
    question_2,
    top_k=3,
)


print()
print("=" * 70)
print("QUESTION 1 RESULTS")
print("=" * 70)

for result in results_1:
    print(
        result["rank"],
        result["chunk_id"],
        result["paper_id"],
        result["section"],
        result["chunk_type"],
        result["page_start"],
        result["page_end"],
        f"{result['distance']:.8f}",
    )


print()
print("=" * 70)
print("QUESTION 2 RESULTS")
print("=" * 70)

for result in results_2:
    print(
        result["rank"],
        result["chunk_id"],
        result["paper_id"],
        result["section"],
        result["chunk_type"],
        result["page_start"],
        result["page_end"],
        f"{result['distance']:.8f}",
    )


ids_1 = [
    result["chunk_id"]
    for result in results_1
]

ids_2 = [
    result["chunk_id"]
    for result in results_2
]

distances_1 = [
    round(result["distance"], 8)
    for result in results_1
]

distances_2 = [
    round(result["distance"], 8)
    for result in results_2
]


print()
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)

if np.allclose(
    embedding_1,
    embedding_2,
    atol=1e-7,
):
    print(
        "[FAIL] Different questions produced "
        "the same embedding."
    )

elif (
    ids_1 == ids_2
    and distances_1 == distances_2
):
    print(
        "[FAIL] ChromaDB returned exactly "
        "the same results and distances."
    )

else:
    print(
        "[PASS] Query embeddings and retrieval "
        "results are different."
    )