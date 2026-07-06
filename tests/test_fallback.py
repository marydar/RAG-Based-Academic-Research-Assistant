import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fallback_service import FallbackService


def main():

    service = FallbackService()

    while True:

        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        result = service.answer(question)

        print("\n==============================")
        print("ANSWER\n")
        print(result["answer"])

        if result["used_fallback"]:
            print("\nFallback: YES")
        else:
            print("\nFallback: NO")

            print("\nSources:")

            for source in result["sources"]:
                print(
                    f"- {source['paper']} | "
                    f"{source['section']} | "
                    f"Pages {source['pages']}"
                )

        print("==============================\n")


if __name__ == "__main__":
    main()