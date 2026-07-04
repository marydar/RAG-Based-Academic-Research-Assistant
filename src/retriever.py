"""Simple two-stage RAG retriever for DETR papers.

Pipeline:
1. Reject clearly out-of-domain questions.
2. Embed the question.
3. Search all papers broadly.
4. Choose the most relevant papers from the broad results.
5. Search again inside selected papers.
6. Rerank and print final evidence.
"""

import math
import re
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_service import EmbeddingService
from src.vector_store import COLLECTION_NAME, DB_PATH


TOP_K = 5
BROAD_TOP_K = 30

MAX_DISTANCE = 0.19
ROUTING_DISTANCE = 0.23
PAPER_SCORE_MARGIN = 0.035
MAX_SELECTED_PAPERS = 4


DOMAIN_PATTERNS = [
    r"\bdetr\b",
    r"\bdeformable\b",
    r"\bconditional\b",
    r"\bdab\b",
    r"\bobject detection\b",
    r"\bobject queries?\b",
    r"\bcoco\b",
    r"\bap\b",
    r"\bap50\b",
    r"\bap75\b",
    r"\bbounding boxes?\b",
    r"\bhungarian\b",
    r"\bbipartite\b",
    r"\bnms\b",
    r"\bpanoptic\b",
    r"\bconvergence\b",
    r"\btraining schedule\b",
    r"\bcross-attention\b",
    r"\bdynamic anchor\b",
    r"\banchor boxes?\b",

    r"دتر",
    r"تشخیص\s+شی",
    r"تشخیص\s+اشیا",
    r"کوئری",
    r"کاندیشنال",
    r"دیفورمبل",
    r"همگرایی",
    r"انکر",
    r"باکس",
]


NUMERIC_PATTERNS = [
    r"\bap\b",
    r"\bap50\b",
    r"\bap75\b",
    r"\bepochs?\b",
    r"\bscore\b",
    r"\bresult\b",
    r"\bperformance\b",
    r"\btable\b",
    r"\bhow many\b",
    r"\bwhat value\b",
    r"چند",
    r"مقدار",
    r"نتیجه",
    r"جدول",
]


SECTION_PENALTY = {
    "methodology": 0.000,
    "introduction": 0.005,
    "abstract": 0.010,
    "experiments": 0.025,
    "results": 0.025,
    "related_work": 0.030,
    "conclusion": 0.035,
    "appendix": 0.050,
    "references": 1.000,
}


def matches_any(text, patterns):
    """Return True if text matches at least one regex pattern."""

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def has_domain_signal(question):
    """Reject questions that are clearly outside the DETR papers."""

    return matches_any(
        question.lower(),
        DOMAIN_PATTERNS,
    )


def is_numeric_or_table_question(question):
    """Detect questions that should prefer tables/experiments."""

    return matches_any(
        question.lower(),
        NUMERIC_PATTERNS,
    )


def open_collection():
    """Open ChromaDB collection."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "ChromaDB was not found. "
            "Run python src/vector_store.py first."
        )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    if collection.count() == 0:
        raise ValueError(
            "The Chroma collection is empty."
        )

    return collection


def make_where_filter(paper_id=None):
    """Create ChromaDB metadata filter."""

    retrieval_filter = {
        "retrieval_enabled": {
            "$eq": True
        }
    }

    if paper_id is None:
        return retrieval_filter

    return {
        "$and": [
            retrieval_filter,
            {
                "paper_id": {
                    "$eq": paper_id
                }
            },
        ]
    }


def query_collection(
    collection,
    query_embedding,
    top_k,
    paper_id=None,
):
    """Query ChromaDB."""

    return collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        where=make_where_filter(paper_id),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


def flatten_results(raw_results):
    """Convert ChromaDB nested output to a simple list of dictionaries."""

    if (
        not raw_results
        or not raw_results.get("ids")
        or not raw_results["ids"][0]
    ):
        return []

    results = []

    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}

        result = {
            "chunk_id": chunk_id,
            "document": documents[index],
            "distance": float(distances[index]),
            "paper_id": metadata.get("paper_id", "unknown"),
            "paper_short_name": metadata.get("paper_short_name", "unknown"),
            "paper_title": metadata.get("paper_title", "unknown"),
            "section": metadata.get("section", "unknown"),
            "chunk_type": metadata.get("chunk_type", "text"),
            "page_start": metadata.get("page_start", "unknown"),
            "page_end": metadata.get("page_end", "unknown"),
        }

        results.append(result)

    return results


def group_by_paper(results):
    """Group broad results by paper_id."""

    groups = {}

    for result in results:
        if result["distance"] > ROUTING_DISTANCE:
            continue

        paper_id = result["paper_id"]

        if paper_id == "unknown":
            continue

        if paper_id not in groups:
            groups[paper_id] = []

        groups[paper_id].append(result)

    for paper_id in groups:
        groups[paper_id].sort(
            key=lambda item: item["distance"]
        )

    return groups


def choose_papers(broad_results):
    """Choose relevant papers from broad retrieval results."""

    if not broad_results:
        return []

    best_distance = min(
        result["distance"]
        for result in broad_results
    )

    if best_distance > MAX_DISTANCE:
        return []

    groups = group_by_paper(broad_results)

    paper_scores = []

    for paper_id, paper_results in groups.items():
        top_results = paper_results[:3]

        distances = [
            result["distance"]
            for result in top_results
        ]

        best = min(distances)
        average = sum(distances) / len(distances)
        hit_count = len(paper_results)

        score = (
            0.65 * best
            + 0.35 * average
            - 0.004 * min(hit_count, 5)
        )

        paper_scores.append(
            {
                "paper_id": paper_id,
                "score": score,
                "best_distance": best,
            }
        )

    if not paper_scores:
        return []

    paper_scores.sort(
        key=lambda item: item["score"]
    )

    best_score = paper_scores[0]["score"]

    selected_papers = []

    for paper in paper_scores:
        if paper["best_distance"] > MAX_DISTANCE:
            continue

        if paper["score"] <= best_score + PAPER_SCORE_MARGIN:
            selected_papers.append(
                paper["paper_id"]
            )

    return selected_papers[:MAX_SELECTED_PAPERS]


def calculate_rerank_score(result, question):
    """Add a small metadata-based rerank score."""

    score = result["distance"]

    section = str(result["section"]).lower()
    chunk_type = str(result["chunk_type"]).lower()

    if is_numeric_or_table_question(question):
        if chunk_type == "table":
            score -= 0.020

        if section in ["experiments", "results"]:
            score -= 0.010

    else:
        score += SECTION_PENALTY.get(section, 0.030)

        if chunk_type == "table":
            score += 0.015

    result["rerank_score"] = score

    return result


def remove_weak_and_duplicate_results(results):
    """Remove high-distance and duplicate chunks."""

    unique = {}

    for result in results:
        if result["distance"] > MAX_DISTANCE:
            continue

        chunk_id = result["chunk_id"]

        if (
            chunk_id not in unique
            or result["distance"] < unique[chunk_id]["distance"]
        ):
            unique[chunk_id] = result

    return list(unique.values())


def balance_results(results, selected_papers, top_k):
    """Try to keep at least one result from each selected paper."""

    if len(selected_papers) <= 1:
        return results[:top_k]

    final_results = []
    used_chunk_ids = set()

    for paper_id in selected_papers:
        for result in results:
            if result["paper_id"] == paper_id:
                final_results.append(result)
                used_chunk_ids.add(result["chunk_id"])
                break

    for result in results:
        if len(final_results) >= top_k:
            break

        if result["chunk_id"] in used_chunk_ids:
            continue

        final_results.append(result)
        used_chunk_ids.add(result["chunk_id"])

    final_results.sort(
        key=lambda item: item["rerank_score"]
    )

    return final_results[:top_k]


def search(question, collection, embedding_service, top_k=TOP_K):
    """Run two-stage RAG retrieval."""

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not has_domain_signal(question):
        return []

    query_embedding = embedding_service.encode_query(
        question
    )

    # Stage 1: broad retrieval over all papers
    raw_broad_results = query_collection(
        collection=collection,
        query_embedding=query_embedding,
        top_k=BROAD_TOP_K,
        paper_id=None,
    )

    broad_results = flatten_results(
        raw_broad_results
    )

    selected_papers = choose_papers(
        broad_results
    )

    if not selected_papers:
        return []

    # Stage 2: focused retrieval over selected papers
    final_candidates = []

    results_per_paper = max(
        2,
        math.ceil(top_k / len(selected_papers)),
    )

    for paper_id in selected_papers:
        raw_results = query_collection(
            collection=collection,
            query_embedding=query_embedding,
            top_k=results_per_paper,
            paper_id=paper_id,
        )

        final_candidates.extend(
            flatten_results(raw_results)
        )

    for result in final_candidates:
        calculate_rerank_score(
            result,
            question,
        )

    final_candidates = remove_weak_and_duplicate_results(
        final_candidates
    )

    final_candidates.sort(
        key=lambda item: item["rerank_score"]
    )

    final_results = balance_results(
        results=final_candidates,
        selected_papers=selected_papers,
        top_k=top_k,
    )

    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank

    return final_results


def format_pages(result):
    """Format page numbers."""

    page_start = str(result["page_start"])
    page_end = str(result["page_end"])

    if page_start == page_end:
        return page_start

    return f"{page_start} - {page_end}"


def print_results(results):
    """Print retrieved chunks."""

    if not results:
        print()
        print(
            "No sufficiently relevant evidence "
            "was found in the papers."
        )
        return

    print()
    print(f"Retrieved chunks: {len(results)}")

    for result in results:
        print()
        print("=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(
            f"Paper: {result['paper_id']} - "
            f"{result['paper_short_name']}"
        )
        print(f"Section: {result['section']}")
        print(f"Content type: {result['chunk_type']}")
        print(f"Pages: {format_pages(result)}")
        print(f"Cosine distance: {result['distance']:.6f}")
        print(f"Rerank score: {result['rerank_score']:.6f}")
        print("-" * 80)
        print(result["document"])

    print()
    print("=" * 80)


def main():
    """Run the interactive retriever."""

    try:
        collection = open_collection()
        embedding_service = EmbeddingService()

    except Exception as error:
        print(f"Startup error: {error}")
        return

    print("Simple two-stage RAG Retriever is ready.")
    print("Type exit to close the program.")

    while True:
        print()

        question = input("Ask your question:\n> ").strip()

        if question.lower() == "exit":
            print("Retriever closed.")
            break

        try:
            results = search(
                question=question,
                collection=collection,
                embedding_service=embedding_service,
                top_k=TOP_K,
            )

            print_results(results)

        except Exception as error:
            print(f"Retrieval error: {error}")


if __name__ == "__main__":
    main()
