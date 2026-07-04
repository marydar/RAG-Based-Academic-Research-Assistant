import json
from pathlib import Path


PAPER_IDS = ["P1", "P2", "P3", "P4"]
EXTRACTED_DIR = Path("data/extracted")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def short_value(value, max_length=300):
    text = str(value).replace("\n", " ")

    if len(text) > max_length:
        return text[:max_length] + "..."

    return text


def print_dict_structure(data, level=0, max_level=3):
    if level > max_level:
        return

    indent = "    " * level

    if isinstance(data, dict):
        for key, value in data.items():
            value_type = type(value).__name__

            if isinstance(value, list):
                print(
                    f"{indent}- {key}: list "
                    f"(length={len(value)})"
                )

                if value and level < max_level:
                    first_item = value[0]

                    print(
                        f"{indent}    first item type: "
                        f"{type(first_item).__name__}"
                    )

                    if isinstance(first_item, dict):
                        print(
                            f"{indent}    first item keys: "
                            f"{list(first_item.keys())}"
                        )

            elif isinstance(value, dict):
                print(
                    f"{indent}- {key}: dict "
                    f"(keys={list(value.keys())[:20]})"
                )

                if level < max_level:
                    print_dict_structure(
                        value,
                        level + 1,
                        max_level,
                    )

            else:
                print(
                    f"{indent}- {key}: {value_type} = "
                    f"{short_value(value)}"
                )


def find_key_recursively(data, target_key, current_path="root"):
    results = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}"

            if key == target_key:
                results.append(
                    {
                        "path": new_path,
                        "type": type(value).__name__,
                        "length": (
                            len(value)
                            if isinstance(value, (list, dict, str))
                            else None
                        ),
                        "sample": short_value(value),
                    }
                )

            results.extend(
                find_key_recursively(
                    value,
                    target_key,
                    new_path,
                )
            )

    elif isinstance(data, list):
        for index, item in enumerate(data[:20]):
            new_path = f"{current_path}[{index}]"

            results.extend(
                find_key_recursively(
                    item,
                    target_key,
                    new_path,
                )
            )

    return results


def find_pdf_files():
    print("=" * 80)
    print("PDF FILES FOUND IN PROJECT")
    print("=" * 80)

    pdf_files = list(Path(".").rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files were found.")
        return

    for pdf_path in pdf_files:
        try:
            size = pdf_path.stat().st_size
        except OSError:
            size = 0

        print(
            f"{pdf_path} | "
            f"size={size:,} bytes"
        )


find_pdf_files()
print()


for paper_id in PAPER_IDS:
    print("=" * 80)
    print(f"INSPECTING {paper_id}")
    print("=" * 80)

    paper_dir = EXTRACTED_DIR / paper_id
    metadata_path = paper_dir / "metadata.json"
    document_path = paper_dir / "document.json"
    markdown_path = paper_dir / "document.md"

    if not metadata_path.exists():
        print("ERROR: metadata.json was not found.")
        continue

    if not document_path.exists():
        print("ERROR: document.json was not found.")
        continue

    metadata = load_json(metadata_path)
    document = load_json(document_path)

    print()
    print("METADATA.JSON CONTENT")
    print("-" * 80)

    print(json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("DOCUMENT.JSON TOP-LEVEL KEYS")
    print("-" * 80)

    if isinstance(document, dict):
        print(list(document.keys()))
    else:
        print(
            "document.json root type:",
            type(document).__name__,
        )

    print()
    print("DOCUMENT.JSON STRUCTURE")
    print("-" * 80)

    print_dict_structure(
        document,
        level=0,
        max_level=2,
    )

    print()
    print("IMPORTANT KEYS SEARCH")
    print("-" * 80)

    important_keys = [
        "pages",
        "page",
        "page_number",
        "page_no",
        "texts",
        "text",
        "tables",
        "pictures",
        "body",
        "structured_document",
        "origin",
        "filename",
        "mimetype",
        "page_count",
        "num_pages",
    ]

    for key in important_keys:
        results = find_key_recursively(
            document,
            key,
        )

        if results:
            print(f"\nKEY: {key}")

            for result in results[:10]:
                print(
                    f"  path={result['path']} | "
                    f"type={result['type']} | "
                    f"length={result['length']}"
                )

                print(
                    f"  sample={result['sample'][:300]}"
                )
        else:
            print(f"\nKEY: {key} -> NOT FOUND")

    print()
    print("MARKDOWN CHECK")
    print("-" * 80)

    if markdown_path.exists():
        markdown_text = markdown_path.read_text(
            encoding="utf-8"
        )

        print(
            "Markdown characters:",
            len(markdown_text),
        )

        print(
            "Markdown beginning:"
        )

        print(
            markdown_text[:1000]
            .replace("\n", " ")
        )
    else:
        print("document.md was not found.")

    print()