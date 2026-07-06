"""
Handles questions that cannot be answered from the retrieved papers.
"""

from src.llm_service import LLMService
from src.rag_service import RAGService


class FallbackService:
    """
    Decides whether to use the RAG answer or fall back
    to the language model's general knowledge.
    """

    def __init__(self):

        self.rag = RAGService()
        self.llm = LLMService()

    def answer(self, question: str):
        """
        Answer a question using RAG if possible.
        Otherwise, use the language model directly.
        """

        rag_result = self.rag.answer(question)

        # No evidence found
        if not rag_result["chunks"]:

            prompt = f"""
You are a helpful AI assistant.

Answer the following question using your own general knowledge.

Question:
{question}

Answer:
"""

            general_answer = self.llm.chat(prompt)

            return {
                "answer":
                    "The requested information was not found in the selected papers.\n\n"
                    "However, using the language model, the general answer is:\n\n"
                    + general_answer,

                "sources": [],

                "chunks": [],

                "used_fallback": True,
            }

        # Normal RAG answer
        rag_result["used_fallback"] = False

        return rag_result