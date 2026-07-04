import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pages(paper_id):
    path = (
        PROJECT_ROOT
        / "data"
        / "extracted"
        / paper_id
        / "document.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["document"]["pages"]


def main():
    for paper_id in ["P1", "P2", "P3", "P4"]:
        pages = load_pages(paper_id)

        print("\n" + "#" * 80)
        print(f"PAPER: {paper_id}")
        print("#" * 80)

        table_found = False

        for page in pages:
            page_number = page["page_number"]
            structured = page.get("structured_document", {})
            tables = structured.get("tables", [])

            if not tables:
                continue

            table_found = True

            print(
                f"\nPage {page_number} | "
                f"Table count: {len(tables)}"
            )

            for index, table in enumerate(tables):
                print("\n" + "=" * 70)
                print(f"TABLE {index}")
                print("Keys:", list(table.keys()))

                print(
                    json.dumps(
                        table,
                        ensure_ascii=False,
                        indent=2,
                    )[:4000]
                )

                # فقط اولین جدول هر مقاله کافی است.
                break

            # فقط اولین صفحه دارای جدول هر مقاله
            break

        if not table_found:
            print("No tables were found.")


if __name__ == "__main__":
    main()