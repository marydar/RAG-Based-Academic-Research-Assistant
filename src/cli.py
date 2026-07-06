"""
Command Line Interface for the RAG Academic Research Assistant.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.agent import Agent


def print_sources(sources):
    """Print retrieved sources."""

    if not sources:
        return

    print("\nSources:")

    for source in sources:

        pages = source["pages"]

        if "-" in pages:
            start, end = pages.split("-")

            if start == end:
                pages = f"Page {start}"
            else:
                pages = f"Pages {start}-{end}"

        print(
            f" • {source['paper']} | "
            f"{source['section'].title()} | "
            f"{pages}"
        )


def main():

    print("=" * 60)
    print("Academic Research Assistant")
    print("Type 'exit' to quit.")
    print("=" * 60)

    agent = Agent()

    while True:

        question = input("Ask your question:\n> ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            result = agent.answer(question)

            print()
            print(result["answer"])

            print_sources(result.get("sources", []))

        except Exception as error:

            print(f"\nError: {error}")


if __name__ == "__main__":
    main()