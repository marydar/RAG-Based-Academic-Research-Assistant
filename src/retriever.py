import sys
from pathlib import Path

import chromadb


# برای اینکه فایل هم مستقیم و هم از tests قابل Import باشد.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_service import EmbeddingService
from src.vector_store import COLLECTION_NAME, DB_PATH


DEFAULT_TOP_K = 5


class SemanticRetriever:
    """Chunkهای مرتبط را از ChromaDB بازیابی می‌کند."""

    def __init__(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(
                "ChromaDB was not found. "
                "First run: python src\\vector_store.py"
            )

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not open collection "
                f"'{COLLECTION_NAME}': {error}"
            ) from error

        if self.collection.count() == 0:
            raise ValueError(
                "The Chroma collection is empty."
            )

        self.embedding_service = EmbeddingService()

    def search(self, question, top_k=DEFAULT_TOP_K):
        """برای سؤال کاربر، مرتبط‌ترین Chunkها را برمی‌گرداند."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(
                "top_k must be a positive integer."
            )

        # تولید بردار 384 بعدی سؤال
        query_embedding = (
            self.embedding_service.encode_query(
                question.strip()
            )
        )

        # جست‌وجوی معنایی در ChromaDB
        raw_results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            where={
                "retrieval_enabled": True
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        results = []

        for index in range(len(ids)):
            metadata = metadatas[index]

            result = {
                "rank": index + 1,
                "chunk_id": ids[index],
                "document": documents[index],
                "paper_id": metadata.get(
                    "paper_id",
                    "unknown",
                ),
                "paper_title": metadata.get(
                    "paper_title",
                    "unknown",
                ),
                "section": metadata.get(
                    "section",
                    "unknown",
                ),
                "page_start": metadata.get(
                    "page_start",
                    "unknown",
                ),
                "page_end": metadata.get(
                    "page_end",
                    "unknown",
                ),
                "retrieval_enabled": metadata.get(
                    "retrieval_enabled",
                    True,
                ),
                "distance": float(
                    distances[index]
                ),
            }

            results.append(result)

        return results


def format_pages(result):
    """شماره صفحه یا بازه صفحات را آماده نمایش می‌کند."""

    page_start = str(result["page_start"])
    page_end = str(result["page_end"])

    if page_start == page_end:
        return page_start

    return f"{page_start} - {page_end}"


def print_results(results):
    """نتایج بازیابی‌شده را در ترمینال نمایش می‌دهد."""

    if not results:
        print("No relevant chunks were found.")
        return

    print()
    print(f"Retrieved chunks: {len(results)}")

    for result in results:
        print()
        print("=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(
            f"Paper: {result['paper_id']} - "
            f"{result['paper_title']}"
        )
        print(f"Section: {result['section']}")
        print(f"Pages: {format_pages(result)}")
        print(
            f"Cosine distance: "
            f"{result['distance']:.6f}"
        )
        print("-" * 80)
        print(result["document"])

    print()
    print("=" * 80)


def main():
    """اجرای ساده Retriever از طریق ترمینال."""

    try:
        retriever = SemanticRetriever()
    except Exception as error:
        print(f"Startup error: {error}")
        return

    print("Semantic Retriever is ready.")
    print("Type exit to close the program.")

    while True:
        print()
        question = input("Ask your question:\n> ").strip()

        if question.lower() == "exit":
            print("Retriever closed.")
            break

        try:
            results = retriever.search(
                question=question,
                top_k=DEFAULT_TOP_K,
            )

            print_results(results)

        except Exception as error:
            print(f"Retrieval error: {error}")


if __name__ == "__main__":
    main()