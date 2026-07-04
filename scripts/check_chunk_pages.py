import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "all_chunks.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "chunk_page_report.json"
)


def get_value(chunk, *keys, default=None):
    """مقدار را از خود Chunk یا metadata می‌خواند."""

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


def to_page_number(value):
    """شماره صفحه را به عدد صحیح تبدیل می‌کند."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_chunks():
    """فایل Chunkها را می‌خواند."""

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data.get("chunks", []) if isinstance(data, dict) else data

    if not chunks:
        raise ValueError("No chunks were found.")

    return chunks


def main():
    chunks = load_chunks()

    missing_pages = []
    invalid_ranges = []
    wide_ranges = []
    range_groups = defaultdict(list)

    for chunk in chunks:
        chunk_id = str(
            get_value(
                chunk,
                "chunk_id",
                "id",
                default="unknown",
            )
        )

        paper_id = str(
            get_value(
                chunk,
                "paper_id",
                default="unknown",
            )
        )

        section = str(
            get_value(
                chunk,
                "section",
                "section_name",
                "section_title",
                default="unknown",
            )
        )

        page_start = to_page_number(
            get_value(
                chunk,
                "page_start",
                "start_page",
            )
        )

        page_end = to_page_number(
            get_value(
                chunk,
                "page_end",
                "end_page",
            )
        )

        retrieval_enabled = get_value(
            chunk,
            "retrieval_enabled",
            default=None,
        )

        if retrieval_enabled is None:
            retrieval_enabled = (
                section.lower() != "references"
            )

        if isinstance(retrieval_enabled, str):
            retrieval_enabled = (
                retrieval_enabled.lower()
                not in {"false", "0", "no"}
            )
        else:
            retrieval_enabled = bool(retrieval_enabled)

        item = {
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
            "retrieval_enabled": retrieval_enabled,
        }

        if page_start is None or page_end is None:
            missing_pages.append(item)
            continue

        if page_start > page_end:
            invalid_ranges.append(item)
            continue

        page_count = page_end - page_start + 1
        item["page_count"] = page_count

        if retrieval_enabled and page_count > 3:
            wide_ranges.append(item)

        group_key = (
            paper_id,
            section,
            page_start,
            page_end,
        )

        range_groups[group_key].append(chunk_id)

    repeated_ranges = []

    for key, chunk_ids in range_groups.items():
        paper_id, section, page_start, page_end = key

        if len(chunk_ids) > 1:
            repeated_ranges.append(
                {
                    "paper_id": paper_id,
                    "section": section,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_count": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                }
            )

    repeated_ranges.sort(
        key=lambda item: item["chunk_count"],
        reverse=True,
    )

    report = {
        "total_chunks": len(chunks),
        "missing_page_count": len(missing_pages),
        "invalid_range_count": len(invalid_ranges),
        "wide_range_count": len(wide_ranges),
        "repeated_range_count": len(repeated_ranges),
        "missing_pages": missing_pages,
        "invalid_ranges": invalid_ranges,
        "wide_ranges": wide_ranges,
        "repeated_ranges": repeated_ranges,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks with missing pages: {len(missing_pages)}")
    print(f"Chunks with invalid ranges: {len(invalid_ranges)}")
    print(f"Active chunks spanning over 3 pages: {len(wide_ranges)}")
    print(f"Repeated page ranges: {len(repeated_ranges)}")

    if wide_ranges:
        print("\nWide page ranges:")

        for item in wide_ranges[:10]:
            print(
                f"{item['chunk_id']} | "
                f"{item['paper_id']} | "
                f"{item['section']} | "
                f"{item['page_start']}-{item['page_end']}"
            )

    if repeated_ranges:
        print("\nMost repeated page ranges:")

        for item in repeated_ranges[:10]:
            print(
                f"{item['paper_id']} | "
                f"{item['section']} | "
                f"{item['page_start']}-{item['page_end']} | "
                f"{item['chunk_count']} chunks"
            )

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()