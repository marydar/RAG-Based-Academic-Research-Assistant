"""Create page-aware text and table chunks."""

import json
import re
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

PAPERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "papers.json"
)

SECTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "sections"
)

CHUNKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "chunks"
)


CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

SKIPPED_SECTIONS = {
    "front_matter",
}

TOKENIZER = (
    tiktoken.get_encoding(
        "cl100k_base"
    )
)


def load_json(path):
    """Load JSON data."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(path, data):
    """Save JSON data."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def count_tokens(text):
    """Count cl100k tokens."""

    return len(
        TOKENIZER.encode(
            text,
            disallowed_special=(),
        )
    )


def split_text(text):
    """Split normal text using token overlap."""

    text = text.strip()

    if not text:
        return []

    token_ids = TOKENIZER.encode(
        text,
        disallowed_special=(),
    )

    if len(token_ids) <= CHUNK_SIZE:
        return [
            {
                "text": text,
                "token_count": len(
                    token_ids
                ),
            }
        ]

    step_size = (
        CHUNK_SIZE
        - CHUNK_OVERLAP
    )

    parts = []
    start = 0

    while start < len(
        token_ids
    ):
        end = min(
            start + CHUNK_SIZE,
            len(token_ids),
        )

        part_text = (
            TOKENIZER.decode(
                token_ids[
                    start:end
                ]
            ).strip()
        )

        if part_text:
            parts.append(
                {
                    "text": (
                        part_text
                    ),
                    "token_count": (
                        end - start
                    ),
                }
            )

        if end >= len(
            token_ids
        ):
            break

        start += step_size

    return parts


def is_table_line(line):
    """Check whether a line belongs to a Markdown table."""

    stripped = line.strip()

    if not stripped:
        return False

    return (
        stripped.count("|")
        >= 2
    )


def is_separator_line(line):
    """Check whether a table line is a Markdown separator."""

    stripped = (
        line.strip()
        .strip("|")
        .strip()
    )

    if not stripped:
        return False

    cells = [
        cell.strip()
        for cell in stripped.split(
            "|"
        )
    ]

    if not cells:
        return False

    return all(
        re.fullmatch(
            r":?-{3,}:?",
            cell,
        )
        is not None
        for cell in cells
    )


def separate_content(text):
    """Separate normal text and tables in original order."""

    segments = []

    text_lines = []
    table_lines = []

    def save_text():
        nonlocal text_lines

        value = "\n".join(
            text_lines
        ).strip()

        if value:
            segments.append(
                {
                    "type": "text",
                    "text": value,
                }
            )

        text_lines = []

    def save_table():
        nonlocal table_lines

        value = "\n".join(
            table_lines
        ).strip()

        if value:
            segments.append(
                {
                    "type": "table",
                    "text": value,
                }
            )

        table_lines = []

    for line in text.splitlines():
        if is_table_line(line):
            if text_lines:
                save_text()

            table_lines.append(
                line
            )

        else:
            if table_lines:
                save_table()

            text_lines.append(
                line
            )

    if text_lines:
        save_text()

    if table_lines:
        save_table()

    return segments


def split_table(table_text):
    """Split a large Markdown table and repeat its header."""

    if (
        count_tokens(
            table_text
        )
        <= CHUNK_SIZE
    ):
        return [
            table_text.strip()
        ]

    lines = [
        line
        for line
        in table_text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return [
            item["text"]
            for item in split_text(
                table_text
            )
        ]

    if (
        len(lines) >= 2
        and is_separator_line(
            lines[1]
        )
    ):
        header_lines = (
            lines[:2]
        )
        data_rows = lines[2:]

    else:
        header_lines = [
            lines[0]
        ]
        data_rows = lines[1:]

    table_parts = []
    current_lines = (
        header_lines.copy()
    )

    for row in data_rows:
        candidate_lines = (
            current_lines
            + [row]
        )

        candidate_text = (
            "\n".join(
                candidate_lines
            )
        )

        if (
            count_tokens(
                candidate_text
            )
            > CHUNK_SIZE
            and len(
                current_lines
            )
            > len(
                header_lines
            )
        ):
            table_parts.append(
                "\n".join(
                    current_lines
                ).strip()
            )

            current_lines = (
                header_lines.copy()
            )

        current_lines.append(
            row
        )

        current_text = (
            "\n".join(
                current_lines
            )
        )

        if (
            count_tokens(
                current_text
            )
            > CHUNK_SIZE
        ):
            fallback_parts = (
                split_text(
                    current_text
                )
            )

            table_parts.extend(
                part["text"]
                for part
                in fallback_parts
            )

            current_lines = (
                header_lines.copy()
            )

    if (
        len(current_lines)
        > len(header_lines)
    ):
        table_parts.append(
            "\n".join(
                current_lines
            ).strip()
        )

    return [
        part
        for part
        in table_parts
        if part.strip()
    ]


def chunk_paper(paper):
    """Create chunks for one paper."""

    paper_id = paper[
        "paper_id"
    ]

    sections_path = (
        SECTIONS_DIR
        / paper_id
        / "sections.json"
    )

    if not sections_path.exists():
        raise FileNotFoundError(
            f"Sections file not found: "
            f"{sections_path}"
        )

    section_data = load_json(
        sections_path
    )

    sections = section_data.get(
        "sections",
        [],
    )

    if not sections:
        raise ValueError(
            f"No sections found "
            f"for {paper_id}."
        )

    paper_chunks = []

    for section in sections:
        section_name = section[
            "canonical_name"
        ]

        if (
            section_name
            in SKIPPED_SECTIONS
        ):
            continue

        blocks = section.get(
            "blocks",
            [],
        )

        if not blocks:
            blocks = [
                {
                    "page_number": (
                        section[
                            "page_start"
                        ]
                    ),
                    "text": (
                        section[
                            "text"
                        ]
                    ),
                }
            ]

        prepared_chunks = []

        for block in blocks:
            page_number = int(
                block[
                    "page_number"
                ]
            )

            block_text = block.get(
                "text",
                "",
            ).strip()

            if not block_text:
                continue

            segments = separate_content(
                block_text
            )

            for segment in segments:
                if (
                    segment["type"]
                    == "table"
                ):
                    table_parts = (
                        split_table(
                            segment[
                                "text"
                            ]
                        )
                    )

                    for part in table_parts:
                        prepared_chunks.append(
                            {
                                "chunk_type": (
                                    "table"
                                ),
                                "page_number": (
                                    page_number
                                ),
                                "text": part,
                                "token_count": (
                                    count_tokens(
                                        part
                                    )
                                ),
                            }
                        )

                else:
                    text_parts = (
                        split_text(
                            segment[
                                "text"
                            ]
                        )
                    )

                    for part in text_parts:
                        prepared_chunks.append(
                            {
                                "chunk_type": (
                                    "text"
                                ),
                                "page_number": (
                                    page_number
                                ),
                                "text": (
                                    part[
                                        "text"
                                    ]
                                ),
                                "token_count": (
                                    part[
                                        "token_count"
                                    ]
                                ),
                            }
                        )

        retrieval_enabled = (
            section_name
            != "references"
        )

        total_chunks = len(
            prepared_chunks
        )

        for index, prepared in enumerate(
            prepared_chunks,
            start=1,
        ):
            chunk_id = (
                f"{section['section_id']}"
                f"_chunk_{index:03d}"
            )

            page_number = prepared[
                "page_number"
            ]

            embedding_text = (
                f"Paper: "
                f"{paper['title']}\n"
                f"Paper short name: "
                f"{paper['short_name']}\n"
                f"Section: "
                f"{section['display_name']}\n"
                f"Content type: "
                f"{prepared['chunk_type']}\n"
                f"Page: {page_number}\n\n"
                f"{prepared['text']}"
            )

            paper_chunks.append(
                {
                    "chunk_id": (
                        chunk_id
                    ),
                    "chunk_type": (
                        prepared[
                            "chunk_type"
                        ]
                    ),
                    "paper_id": (
                        paper_id
                    ),
                    "paper_title": (
                        paper[
                            "title"
                        ]
                    ),
                    "paper_short_name": (
                        paper[
                            "short_name"
                        ]
                    ),
                    "section_id": (
                        section[
                            "section_id"
                        ]
                    ),
                    "section": (
                        section_name
                    ),
                    "section_name": (
                        section_name
                    ),
                    "section_title": (
                        section[
                            "display_name"
                        ]
                    ),
                    "page_start": (
                        page_number
                    ),
                    "page_end": (
                        page_number
                    ),
                    "chunk_index": (
                        index
                    ),
                    "chunks_in_section": (
                        total_chunks
                    ),
                    "token_count": (
                        prepared[
                            "token_count"
                        ]
                    ),
                    "retrieval_enabled": (
                        retrieval_enabled
                    ),
                    "include_in_default_retrieval": (
                        retrieval_enabled
                    ),
                    "text": (
                        prepared[
                            "text"
                        ]
                    ),
                    "embedding_text": (
                        embedding_text
                    ),
                }
            )

    return paper_chunks


def process_all_papers():
    """Create chunks for all papers."""

    papers_data = load_json(
        PAPERS_PATH
    )

    if isinstance(
        papers_data,
        dict,
    ):
        papers = papers_data.get(
            "papers",
            [],
        )
    else:
        papers = papers_data

    if not isinstance(
        papers,
        list,
    ) or not papers:
        raise ValueError(
            "papers.json must contain "
            "a non-empty list."
        )

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_chunks = []
    summary = []

    print(
        f"Found {len(papers)} papers."
    )

    for paper in papers:
        paper_id = paper[
            "paper_id"
        ]

        print()
        print(
            f"Chunking {paper_id}: "
            f"{paper['short_name']}"
        )

        chunks = chunk_paper(
            paper
        )

        if not chunks:
            raise ValueError(
                f"No chunks were created "
                f"for {paper_id}."
            )

        output_dir = (
            CHUNKS_DIR
            / paper_id
        )

        save_json(
            output_dir
            / "chunks.json",
            {
                "paper": paper,
                "chunking_config": {
                    "tokenizer": (
                        "cl100k_base"
                    ),
                    "chunk_size": (
                        CHUNK_SIZE
                    ),
                    "chunk_overlap": (
                        CHUNK_OVERLAP
                    ),
                    "skipped_sections": (
                        sorted(
                            SKIPPED_SECTIONS
                        )
                    ),
                },
                "chunk_count": (
                    len(chunks)
                ),
                "chunks": chunks,
            },
        )

        all_chunks.extend(
            chunks
        )

        text_count = sum(
            chunk[
                "chunk_type"
            ]
            == "text"
            for chunk in chunks
        )

        table_count = sum(
            chunk[
                "chunk_type"
            ]
            == "table"
            for chunk in chunks
        )

        enabled_count = sum(
            chunk[
                "retrieval_enabled"
            ]
            for chunk in chunks
        )

        reference_count = sum(
            chunk["section"]
            == "references"
            for chunk in chunks
        )

        print(
            f"  Chunks: "
            f"{len(chunks)}"
        )

        print(
            f"  Text chunks: "
            f"{text_count}"
        )

        print(
            f"  Table chunks: "
            f"{table_count}"
        )

        print(
            f"  Retrieval enabled: "
            f"{enabled_count}"
        )

        print(
            f"  Reference chunks disabled: "
            f"{reference_count}"
        )

        summary.append(
            {
                "paper_id": (
                    paper_id
                ),
                "short_name": (
                    paper[
                        "short_name"
                    ]
                ),
                "chunk_count": (
                    len(chunks)
                ),
                "text_chunk_count": (
                    text_count
                ),
                "table_chunk_count": (
                    table_count
                ),
                "retrieval_enabled_count": (
                    enabled_count
                ),
            }
        )

    chunk_ids = [
        chunk[
            "chunk_id"
        ]
        for chunk in all_chunks
    ]

    if len(chunk_ids) != len(
        set(chunk_ids)
    ):
        raise ValueError(
            "Duplicate chunk IDs "
            "were found."
        )

    multi_page_chunks = sum(
        chunk["page_start"]
        != chunk["page_end"]
        for chunk in all_chunks
    )

    save_json(
        CHUNKS_DIR
        / "all_chunks.json",
        all_chunks,
    )

    save_json(
        CHUNKS_DIR
        / "summary.json",
        summary,
    )

    print()
    print("-" * 50)

    print(
        f"Total chunks: "
        f"{len(all_chunks)}"
    )

    print(
        "Retrieval-enabled chunks: "
        f"{sum(c['retrieval_enabled'] for c in all_chunks)}"
    )

    print(
        "Table chunks: "
        f"{sum(c['chunk_type'] == 'table' for c in all_chunks)}"
    )

    print(
        f"Multi-page chunks: "
        f"{multi_page_chunks}"
    )

    print(
        "Largest chunk: "
        f"{max(c['token_count'] for c in all_chunks)} "
        f"cl100k tokens"
    )

    print(
        "Chunking completed "
        "successfully."
    )


if __name__ == "__main__":
    process_all_papers()