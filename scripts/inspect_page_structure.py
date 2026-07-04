import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_page_fields(data, path="root", results=None):
    """تمام فیلدهایی را که نام page دارند پیدا می‌کند."""

    if results is None:
        results = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}"

            if "page" in key.lower():
                results.append(
                    (current_path, str(value)[:150])
                )

            find_page_fields(
                value,
                current_path,
                results,
            )

    elif isinstance(data, list):
        for index, item in enumerate(data[:20]):
            find_page_fields(
                item,
                f"{path}[{index}]",
                results,
            )

    return results


def get_first_section(data):
    """اولین Section را از ساختارهای رایج پیدا می‌کند."""

    if isinstance(data, list):
        return data[0] if data else None

    if isinstance(data, dict):
        sections = data.get("sections")

        if isinstance(sections, list) and sections:
            return sections[0]

        for value in data.values():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    return value[0]

    return None


def inspect_file(path):
    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    if not path.exists():
        print("FILE NOT FOUND")
        return

    data = load_json(path)

    print("Top-level type:", type(data).__name__)

    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))

    first_section = get_first_section(data)

    if isinstance(first_section, dict):
        print("First section keys:", list(first_section.keys()))

        print("\nFirst section sample:")

        sample = json.dumps(
            first_section,
            ensure_ascii=False,
            indent=2,
        )

        print(sample[:3000])

    page_fields = find_page_fields(data)

    print("\nPage-related fields found:", len(page_fields))

    for field_path, value in page_fields[:30]:
        print(f"{field_path} = {value}")


def main():
    for paper_id in ["P1", "P2", "P3", "P4"]:
        sections_path = (
            PROJECT_ROOT
            / "data"
            / "sections"
            / paper_id
            / "sections.json"
        )

        document_path = (
            PROJECT_ROOT
            / "data"
            / "extracted"
            / paper_id
            / "document.json"
        )

        print("\n" + "#" * 80)
        print(f"PAPER: {paper_id}")
        print("#" * 80)

        inspect_file(sections_path)
        inspect_file(document_path)


if __name__ == "__main__":
    main()