"""
Compare the limitations of all selected papers.

Output:
    data/outputs/limitations_comparison.md
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_service import LLMService
from src.vector_store import load_chunks


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
    / "limitations_comparison.md"
)


class LimitationsComparer:

    TARGET_SECTIONS = {
        "conclusion",
        "discussion",
    }

    def __init__(self):

        self.llm = LLMService()

    def collect_sections(self):
        """
        Collect conclusion/discussion chunks for every paper.
        """

        chunks = load_chunks()

        grouped = {}

        for chunk in chunks:

            section = (
                chunk.get("section", "")
                .lower()
                .strip()
            )

            if section not in self.TARGET_SECTIONS:
                continue

            paper = chunk.get(
                "paper_short_name",
                chunk.get("paper_id", "Unknown"),
            )

            grouped.setdefault(
                paper,
                [],
            ).append(chunk["text"])

        if not grouped:
            raise RuntimeError(
                "No conclusion sections were found."
            )

        return grouped

    def build_context(self, grouped):
        """
        Build context for the LLM.
        """

        context = []

        for paper in sorted(grouped):

            context.append(f"# {paper}")

            context.extend(grouped[paper])

            context.append("")

        return "\n\n".join(context)

    def build_prompt(self, context):
        """
        Build prompt.
        """

        return f"""
    You are an expert researcher writing a comparison report of several object detection papers.

    Using ONLY the provided context, analyze every paper individually and then compare them.

    Generate a well-structured Markdown report with the following sections.

    # 1. Overview

    Briefly introduce the purpose of comparing these methods.

    # 2. Comparison Table

    Create a markdown table with these columns:

    | Paper | Main Idea | Key Limitations | Reported Strengths | Future Work |

    Fill every cell.

    # 3. Limitation Comparison

    Create another markdown table.

    | Aspect | DETR | Deformable DETR | Conditional DETR | DAB-DETR |

    Compare the papers according to:

    - Training convergence
    - Small object detection
    - Computational complexity
    - Robustness
    - Scalability
    - Generalization ability

    If an aspect is not explicitly mentioned, write "Not discussed".

    # 4. Evolution of the Methods

    Explain how each newer paper attempts to solve the limitations of previous methods.

    For example:

    DETR
    ↓
    Deformable DETR
    ↓
    Conditional DETR
    ↓
    DAB-DETR

    Describe what improvement each paper introduces.

    # 5. Remaining Challenges

    List the challenges that still remain unsolved across all papers.

    # 6. Overall Comparison

    Discuss:

    - Which method appears strongest overall?
    - Which method has the fewest limitations?
    - Which challenges remain open?
    - What future research directions seem most promising?

    Return only clean Markdown.

    Context:

    {context}
    """

    def save(self, text):

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(text)

    def run(self):

        grouped = self.collect_sections()

        context = self.build_context(grouped)

        prompt = self.build_prompt(context)

        report = self.llm.chat(prompt)

        self.save(report)

        return {
            "answer": (
                "Limitations comparison completed successfully.\n\n"
                "Generated file:\n"
                "- data/outputs/limitations_comparison.md"
            ),
            "sources": [],
        }


if __name__ == "__main__":

    tool = LimitationsComparer()

    result = tool.run()

    print(result["answer"])