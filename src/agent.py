"""
Intelligent agent responsible for deciding whether to:

- answer using RAG
- extract experimental results
- compare results
- generate a survey
"""

import json
import re

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_service import LLMService
from src.rag_service import RAGService
from src.fallback_service import FallbackService

from tools.extract_experimental_results import (
    ExperimentalResultsExtractor,
)

from tools.compare_results import (
    ComparisonService,
)

from tools.generate_survey import (
    SurveyGenerator,
)


VALID_ACTIONS = {
    "rag",
    "extract_experimental_results",
    "compare_results",
    "generate_survey",
}

class Agent:

    def __init__(self):

        self.llm = LLMService()

        self.rag = FallbackService()

        self.extractor = ExperimentalResultsExtractor()

        self.comparer = ComparisonService()

        self.survey = SurveyGenerator()
        
    def build_prompt(
        self,
        question: str,
    ) -> str:
        """
        Build the intent-classification prompt.
        """

        return f"""
        You are an AI routing assistant.

        Your task is to decide which action should be executed.

        Available actions:

        1. rag
        Use this when the user is asking a question that should be answered
        from the research papers.

        Examples:
        - What datasets were used?
        - Explain DETR.
        - Which paper introduced deformable attention?
        - Compare Python vs Java.
        - Who invented Python?

        2. extract_experimental_results
        Use ONLY when the user explicitly asks to extract
        experimental results into a file.

        Examples:
        - Extract experimental results.
        - Export the experiment tables.
        - Create results.csv.

        3. compare_results
        Use ONLY when the user wants to compare the extracted
        experimental results.

        Examples:
        - Compare the papers.
        - Which paper achieved the highest AP?
        - Show the comparison chart.
        - Compare experimental performance.

        4. generate_survey
        Use ONLY when the user wants a survey or literature review.

        Examples:
        - Generate a survey.
        - Write a literature review.
        - Summarize all papers.

        Return ONLY valid JSON.

        Example:

        {{
            "action":"rag",
            "confidence":0.95
        }}

        Question:

        {question}
        """
        
    
    def detect_action(
        self,
        question: str,
    ):
        
        response = self.llm.chat(
            self.build_prompt(question)
        )

        # print("RAW RESPONSE:")
        # print(response)
        # print("-" * 50)
        
        # Extract the first JSON object from the response
        match = re.search(r"\{.*\}", response, re.DOTALL)

        if not match:
            return {
                "action": "rag",
                "confidence": 0.0,
            }

        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            return {
                "action": "rag",
                "confidence": 0.0,
            }

        action = result.get("action", "rag")
        confidence = float(result.get("confidence", 0.0))

        if action not in VALID_ACTIONS:
            action = "rag"

        return {
            "action": action,
            "confidence": confidence,
        }
    def execute_action(
        self,
        action,
        question,
    ):
        """
        Execute the selected action.
        """

        if action == "rag":
            rag_result = self.rag.answer(question)

            return {
                "tool": "rag",
                "answer": rag_result["answer"],
                "sources": rag_result["sources"],
                "used_fallback": rag_result["used_fallback"],

                # Hidden debug information
                "debug": {
                    "chunks": rag_result["chunks"]
                }
            }

        elif action == "extract_experimental_results":
            
            try:
                self.extractor.run()

                return {
                    "tool": "extract_experimental_results",
                    "answer": (
                        "Experimental results extracted successfully.\n\n"
                        "Generated file:\n"
                        "- data/outputs/results.csv"
                    ),
                    "sources": [],
                }
            except Exception as e:
                return {
                    "tool": "extract_experimental_results",
                    "answer": f"exctraction failed:\n{e}",
                    "sources": [],
                }

        elif action == "compare_results":

            try:
                self.comparer.run()

                return {
                "tool": "compare_results",
                "answer": (
                    "Comparison completed successfully.\n\n"
                    "Generated files:\n"
                    "- data/outputs/comparison_table.csv\n"
                    "- data/outputs/comparison_chart.png"
                ),
                "sources": [],
            }
            except Exception as e:
                return {
                    "tool": "compare_results",
                    "answer": f"Comparison failed:\n{e}",
                    "sources": [],
                }

        elif action == "generate_survey":

            try:
                self.survey.run()

                return {
                "tool": "generate_survey",
                "answer": (
                    "Mini survey generated successfully.\n\n"
                    "Generated file:\n"
                    "- data/outputs/mini_survey.md"
                ),
                "sources": [],
            }
            except Exception as e:
                return {
                    "tool": "generate_survey",
                    "answer": f"generation failed:\n{e}",
                    "sources": [],
                }

        return self.execute_action(
            "rag",
            question,
        )
        
    def answer(
        self,
        question,
    ):
        """
        Complete agent pipeline.
        """

        decision = self.detect_action(
            question
        )

        action = decision["action"]

        return self.execute_action(
            action,
            question,
        )
        
def main():

    agent = Agent()

    while True:

        question = input("> ")

        if question.lower() == "exit":
            break

        result = agent.answer(
            question
        )

        print()

        print("=" * 70)

        print(
            f"Executed Tool: {result['tool']}"
        )

        print()

        print(result["answer"])

        print("=" * 70)

        print()


if __name__ == "__main__":
    main()