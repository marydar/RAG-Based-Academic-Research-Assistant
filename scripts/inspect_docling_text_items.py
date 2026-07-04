import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    path = (
        PROJECT_ROOT
        / "data"
        / "extracted"
        / "P1"
        / "document.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    pages = data["document"]["pages"]

    # صفحه 8 مقاله DETR، چون بخش Experiments از این صفحه شروع شده
    page = pages[7]

    print("Page number:", page.get("page_number"))
    print("Page keys:", list(page.keys()))

    structured = page.get("structured_document", {})

    print("\nStructured document keys:")
    print(list(structured.keys()))

    texts = structured.get("texts", [])

    print("\nText item count:", len(texts))

    for index, item in enumerate(texts[:5]):
        print("\n" + "=" * 70)
        print("TEXT ITEM:", index)
        print("Keys:", list(item.keys()))
        print(
            json.dumps(
                item,
                ensure_ascii=False,
                indent=2,
            )[:2500]
        )


if __name__ == "__main__":
    main()