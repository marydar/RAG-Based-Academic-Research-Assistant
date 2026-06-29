"""Validate generated paper chunks."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "all_chunks.json"
)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def main():
    """Validate all generated chunks."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks were generated.")

    chunk_ids = set()
    paper_ids = set()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id in chunk_ids:
            raise ValueError(
                f"Duplicate chunk ID: {chunk_id}"
            )

        chunk_ids.add(chunk_id)
        paper_ids.add(chunk["paper_id"])

        if not chunk["text"].strip():
            raise ValueError(
                f"Empty chunk: {chunk_id}"
            )

        if chunk["token_count"] > CHUNK_SIZE:
            raise ValueError(
                f"Chunk exceeds token limit: "
                f"{chunk_id}"
            )

        if chunk["token_count"] <= 0:
            raise ValueError(
                f"Invalid token count: {chunk_id}"
            )

        if (
            chunk["page_end"]
            < chunk["page_start"]
        ):
            raise ValueError(
                f"Invalid page range: {chunk_id}"
            )

    expected_papers = {
        "P1",
        "P2",
        "P3",
        "P4",
    }

    if paper_ids != expected_papers:
        raise ValueError(
            f"Unexpected paper IDs: {paper_ids}"
        )

    print(f"Total chunks: {len(chunks)}")
    print(f"Maximum token count: {max(c['token_count'] for c in chunks)}")
    print(f"Minimum token count: {min(c['token_count'] for c in chunks)}")
    print(f"Papers: {', '.join(sorted(paper_ids))}")
    print("All chunks are valid.")


if __name__ == "__main__":
    main()