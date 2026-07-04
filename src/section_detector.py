"""Detect the main sections of each paper and preserve page information."""

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPERS_PATH = PROJECT_ROOT / "data" / "papers.json"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
SECTIONS_DIR = PROJECT_ROOT / "data" / "sections"


PAGE_PATTERN = re.compile(r"<!--\s*PAGE\s*:?\s*(\d+).*?-->", re.IGNORECASE)
INLINE_ABSTRACT_PATTERN = re.compile(r"^Abstract\s*[:.-]\s*(.+)$", re.IGNORECASE)


SECTION_TITLES = {
    "front_matter": "Front Matter",
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related_work": "Related Work",
    "methodology": "Methodology",
    "experiments": "Experiments",
    "results": "Results",
    "conclusion": "Conclusion",
    "references": "References",
    "appendix": "Appendix",
}


def load_json(path):
    """Load a UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    """Save data as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalize_heading(text):
    """Normalize a Markdown or numbered heading."""

    text = text.strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip(" .:-–—").lower()


def map_heading_to_section(heading):
    """Map a paper heading to one standard section."""

    title = normalize_heading(heading)

    if title == "abstract":
        return "abstract"

    if title in {"introduction", "intro"}:
        return "introduction"

    if title in {
        "related work",
        "related works",
        "previous work",
        "prior work",
        "background",
        "literature review",
    }:
        return "related_work"

    if title in {
        "method",
        "methods",
        "methodology",
        "approach",
        "proposed method",
        "proposed approach",
        "architecture",
        "model architecture",
        "the detr model",
        "deformable detr",
        "conditional detr",
        "dab-detr",
        "dynamic anchor boxes",
    }:
        return "methodology"

    if title in {
        "experiment",
        "experiments",
        "experimental setup",
        "experimental settings",
        "implementation details",
        "training details",
        "evaluation",
        "ablation study",
        "ablation studies",
    }:
        return "experiments"

    if title in {
        "results",
        "experimental results",
        "quantitative results",
        "qualitative results",
        "evaluation results",
        "performance comparison",
    }:
        return "results"

    if title in {
        "conclusion",
        "conclusions",
        "discussion",
        "concluding remarks",
        "future work",
        "conclusion and future work",
        "limitations",
    }:
        return "conclusion"

    if title in {"references", "bibliography"}:
        return "references"

    if title.startswith("appendix"):
        return "appendix"

    return None


def detect_heading(line):
    """Detect only valid section headings."""

    stripped = line.strip()

    if not stripped:
        return None

    if stripped.startswith("#"):
        return map_heading_to_section(stripped)

    numbered_match = re.match(r"^\d+(?:\.\d+)*[.)]?\s+(.+)$", stripped)

    if numbered_match:
        heading_text = numbered_match.group(1)

        if len(heading_text.split()) <= 12:
            return map_heading_to_section(heading_text)

    if len(stripped.split()) <= 6:
        return map_heading_to_section(stripped)

    return None


def create_section(
    paper_id,
    section_name,
    original_heading,
    page_blocks,
    section_number,
):
    """Create a section with separate blocks for each page."""

    blocks = []

    for page_block in page_blocks:
        text = "\n".join(page_block["lines"]).strip()

        if not text:
            continue

        text = re.sub(r"\n{3,}", "\n\n", text)

        blocks.append(
            {
                "page_number": int(page_block["page_number"]),
                "text": text,
            }
        )

    if not blocks:
        return None

    full_text = "\n\n".join(block["text"] for block in blocks)
    section_id = f"{paper_id}_{section_name}_{section_number:02d}"

    return {
        "section_id": section_id,
        "canonical_name": section_name,
        "display_name": SECTION_TITLES[section_name],
        "original_heading": original_heading,
        "page_start": blocks[0]["page_number"],
        "page_end": blocks[-1]["page_number"],
        "word_count": len(full_text.split()),
        "character_count": len(full_text),
        "text": full_text,
        "blocks": blocks,
    }


def detect_paper_sections(paper):
    """Detect sections for one paper."""

    paper_id = paper["paper_id"]
    markdown_path = EXTRACTED_DIR / paper_id / "document.md"

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    markdown = markdown_path.read_text(encoding="utf-8")

    sections = []
    section_counters = {}

    current_page = 1
    current_section = "front_matter"
    current_heading = "Front Matter"
    current_blocks = []

    def add_line(line):
        """Add a line to the current page block."""

        if not current_blocks or current_blocks[-1]["page_number"] != current_page:
            current_blocks.append(
                {
                    "page_number": current_page,
                    "lines": [],
                }
            )

        current_blocks[-1]["lines"].append(line)

    def save_current_section():
        """Save the current section."""

        nonlocal current_blocks

        if not current_blocks:
            return

        section_counters[current_section] = section_counters.get(current_section, 0) + 1

        section = create_section(
            paper_id=paper_id,
            section_name=current_section,
            original_heading=current_heading,
            page_blocks=current_blocks,
            section_number=section_counters[current_section],
        )

        if section is not None:
            sections.append(section)

        current_blocks = []

    for line in markdown.splitlines():
        stripped = line.strip()

        page_match = PAGE_PATTERN.fullmatch(stripped)

        if page_match:
            current_page = int(page_match.group(1))
            continue

        abstract_match = INLINE_ABSTRACT_PATTERN.match(stripped)

        if abstract_match:
            save_current_section()

            current_section = "abstract"
            current_heading = "Abstract"

            abstract_text = abstract_match.group(1).strip()

            if abstract_text:
                add_line(abstract_text)

            continue

        detected_section = detect_heading(line)

        if detected_section:
            save_current_section()

            current_section = detected_section
            current_heading = normalize_heading(line).title()

            continue

        add_line(line)

    save_current_section()

    found_sections = sorted({section["canonical_name"] for section in sections})

    return {
        "paper": paper,
        "source_file": str(markdown_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "section_count": len(sections),
        "found_sections": found_sections,
        "sections": sections,
    }


def create_normalized_markdown(data):
    """Create readable page-aware Markdown."""

    lines = [
        f"# {data['paper']['title']}",
        "",
    ]

    for section in data["sections"]:
        lines.append(
            f"<!-- SECTION: {section['canonical_name']} "
            f"| PAGES: {section['page_start']}-{section['page_end']} -->"
        )

        lines.append(f"## {section['display_name']}")
        lines.append("")

        for block in section["blocks"]:
            lines.append(f"<!-- PAGE {block['page_number']} -->")
            lines.append(block["text"])
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def process_all_papers():
    """Detect sections for all configured papers."""

    papers_data = load_json(PAPERS_PATH)

    if isinstance(papers_data, dict):
        papers = papers_data.get("papers", [])
    else:
        papers = papers_data

    if not isinstance(papers, list) or not papers:
        raise ValueError("papers.json must contain a non-empty list.")

    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    print(f"Found {len(papers)} papers.")

    for paper in papers:
        paper_id = paper["paper_id"]

        print()
        print(f"Processing {paper_id}: {paper['short_name']}")

        data = detect_paper_sections(paper)

        output_dir = SECTIONS_DIR / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)

        save_json(output_dir / "sections.json", data)

        normalized_markdown = create_normalized_markdown(data)
        (output_dir / "normalized.md").write_text(normalized_markdown, encoding="utf-8")

        print(f"  Sections: {data['section_count']}")
        print("  Found: " + ", ".join(data["found_sections"]))

        summary.append(
            {
                "paper_id": paper_id,
                "short_name": paper["short_name"],
                "section_count": data["section_count"],
                "found_sections": data["found_sections"],
            }
        )

    save_json(SECTIONS_DIR / "summary.json", summary)

    print()
    print("Section detection completed successfully.")


if __name__ == "__main__":
    process_all_papers()
