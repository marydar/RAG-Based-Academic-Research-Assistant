"""Validate paper metadata and PDF file paths."""

import json
from pathlib import Path


def main() -> None:
    metadata_path = Path("data/papers.json")

    if not metadata_path.exists():
        raise FileNotFoundError("data/papers.json was not found.")

    with metadata_path.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    if len(papers) != 4:
        raise ValueError(f"Expected 4 papers, but found {len(papers)}.")

    paper_ids = set()

    for paper in papers:
        paper_id = paper["paper_id"]
        pdf_path = Path(paper["file_path"])

        if paper_id in paper_ids:
            raise ValueError(f"Duplicate paper ID: {paper_id}")

        paper_ids.add(paper_id)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found for {paper_id}: {pdf_path}"
            )

        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")

        print(
            f"{paper_id}: {paper['short_name']} — "
            f"{pdf_path.name} — OK"
        )

    print("\nAll four papers and metadata are valid.")


if __name__ == "__main__":
    main()