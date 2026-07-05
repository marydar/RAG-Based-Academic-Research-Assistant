"""
rag_service.py

Connects Retriever + LLM to create a full RAG system.
"""

from typing import List, Dict, Any

from retriever import search
from llm_service import LLMService
from embedding_service import EmbeddingService
# from src.vector_store import open_collection
import chromadb
from vector_store import DB_PATH, COLLECTION_NAME

class RAGService:

    def __init__(self):
        self.llm = LLMService()
        self.embedding = EmbeddingService()

        client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = client.get_collection(
            name=COLLECTION_NAME
        )

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Convert retrieved chunks into a single context string.
        """

        context_parts = []

        for i, chunk in enumerate(chunks, 1):

            text = chunk["document"]
            meta = chunk

            context_parts.append(
                f"""
[Chunk {i}]
Paper: {meta['paper_title']}
Section: {meta['section']}
Pages: {meta['page_start']} - {meta['page_end']}

Text:
{text}
"""
            )

        return "\n".join(context_parts)

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        question → retrieval → prompt → LLM → answer
        """

        # 1. Retrieve relevant chunks
        chunks = search(
            question=question,
            collection=self.collection,
            embedding_service=self.embedding,
            top_k=5,
        )

        # 2. Handle no results (fallback will come later)
        if not chunks:
            return {
                "answer": "No relevant information found in the papers.",
                "sources": [],
                "chunks": [],
            }

        # 3. Build context
        context = self._build_context(chunks)
        print(context)

        # 4. Build final prompt (IMPORTANT PART)
        prompt = f"""
You are an academic research assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

----------------------
CONTEXT:
{context}
----------------------

QUESTION:
{question}

Answer clearly and academically:
"""

        # 5. Call LLM
        response = self.llm.chat(prompt)

        # 6. Build sources
        sources = [
            {
                "paper": c["paper_short_name"],
                "section": c["section"],
                "pages": f"{c['page_start']}-{c['page_end']}",
            }
            for c in chunks
        ]

        return {
            "answer": response,
            "sources": sources,
            "chunks": chunks,
        }


if __name__ == "__main__":

    rag = RAGService()

    print("RAG system ready. Type 'exit' to stop.\n")

    while True:

        q = input("Question: ").strip()

        if q.lower() == "exit":
            break

        result = rag.answer(q)

        print("\nANSWER:\n")
        print(result["answer"])

        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"- {s['paper']} | {s['section']} | Pages {s['pages']}")

        print("\n" + "=" * 50)