"""Split detected paper sections into overlapping token chunks."""

import json
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPERS_PATH = PROJECT_ROOT / "data" / "papers.json"
SECTIONS_DIR = PROJECT_ROOT / "data" / "sections"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Front matter contains authors and affiliations and is not useful for RAG.
SKIPPED_SECTIONS = {"front_matter"}

# A general-purpose tokenizer for counting and splitting tokens.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def load_json(path: Path):
    """Load a UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data):
    """Save data in UTF-8 JSON format."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def split_text(text: str):
    """Split text into overlapping token chunks."""

    token_ids = TOKENIZER.encode(
        text,
        disallowed_special=(),
    )

    if not token_ids:
        return []

    step_size = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []

    start = 0

    while start < len(token_ids):
        end = min(
            start + CHUNK_SIZE,
            len(token_ids),
        )

        current_tokens = token_ids[start:end]

        chunk_text = TOKENIZER.decode(
            current_tokens
        ).strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "token_count": len(current_tokens),
                    "token_start": start,
                    "token_end": end,
                }
            )

        if end == len(token_ids):
            break

        start += step_size

    return chunks


def chunk_paper(paper: dict):
    """Create chunks for every section of one paper."""

    paper_id = paper["paper_id"]

    sections_path = (
        SECTIONS_DIR
        / paper_id
        / "sections.json"
    )

    if not sections_path.exists():
        raise FileNotFoundError(
            f"Sections file not found: {sections_path}"
        )

    section_data = load_json(sections_path)

    paper_chunks = []

    for section in section_data["sections"]:
        section_name = section["canonical_name"]

        if section_name in SKIPPED_SECTIONS:
            continue

        raw_chunks = split_text(section["text"])
        total_chunks = len(raw_chunks)

        for index, raw_chunk in enumerate(
            raw_chunks,
            start=1,
        ):
            chunk_id = (
                f"{section['section_id']}"
                f"_chunk_{index:03d}"
            )

            # Adding paper and section names improves retrieval later.
            embedding_text = (
                f"Paper: {paper['short_name']}\n"
                f"Section: {section['display_name']}\n\n"
                f"{raw_chunk['text']}"
            )

            paper_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "paper_title": paper["title"],
                    "paper_short_name": paper["short_name"],
                    "section_id": section["section_id"],
                    "section_name": section_name,
                    "section_display_name": (
                        section["display_name"]
                    ),
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "chunk_index": index,
                    "chunks_in_section": total_chunks,
                    "token_count": raw_chunk["token_count"],
                    "token_start": raw_chunk["token_start"],
                    "token_end": raw_chunk["token_end"],
                    "include_in_default_retrieval": (
                        section_name != "references"
                    ),
                    "text": raw_chunk["text"],
                    "embedding_text": embedding_text,
                }
            )

    return paper_chunks


def process_all_papers():
    """Create chunks for all selected papers."""

    papers = load_json(PAPERS_PATH)

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_chunks = []
    summary = []

    print(f"Found {len(papers)} papers.")

    for paper in papers:
        paper_id = paper["paper_id"]

        print()
        print(
            f"Chunking {paper_id}: "
            f"{paper['short_name']}"
        )

        chunks = chunk_paper(paper)

        output_dir = CHUNKS_DIR / paper_id

        save_json(
            output_dir / "chunks.json",
            {
                "paper": paper,
                "chunking_config": {
                    "tokenizer": "cl100k_base",
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "step_size": (
                        CHUNK_SIZE - CHUNK_OVERLAP
                    ),
                    "skipped_sections": sorted(
                        SKIPPED_SECTIONS
                    ),
                },
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
        )

        all_chunks.extend(chunks)

        section_names = sorted(
            {
                chunk["section_name"]
                for chunk in chunks
            }
        )

        summary.append(
            {
                "paper_id": paper_id,
                "short_name": paper["short_name"],
                "chunk_count": len(chunks),
                "sections": section_names,
            }
        )

        print(f"  Chunks: {len(chunks)}")
        print(
            "  Sections: "
            + ", ".join(section_names)
        )
        print(
            f"  Output: "
            f"data/chunks/{paper_id}/chunks.json"
        )

    save_json(
        CHUNKS_DIR / "all_chunks.json",
        all_chunks,
    )

    save_json(
        CHUNKS_DIR / "summary.json",
        summary,
    )

    print()
    print("-" * 50)
    print(f"Total chunks: {len(all_chunks)}")
    print("Chunking completed successfully.")


if __name__ == "__main__":
    process_all_papers()