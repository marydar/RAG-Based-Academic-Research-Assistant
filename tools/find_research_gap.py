"""
Analyze the conclusion sections of all papers and identify:

- Limitations
- Open problems
- Future research directions

Output:
    data/outputs/research_gap.md
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.vector_store import load_chunks
from src.llm_service import LLMService

import json


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
    / "research_gap.md"
)


class ResearchGapFinder:
    """
    Analyze the conclusion sections of every paper.
    """

    TARGET_SECTIONS = {
        "conclusion",
        "discussion",
    }

    def __init__(self):

        self.llm = LLMService()


    def collect_conclusions(self):
        """
        Collect conclusion/discussion chunks from every paper.
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
            ).append(chunk)

        if not grouped:
            raise RuntimeError(
                "No conclusion sections were found."
            )

        return grouped

    def build_context(self, grouped_chunks):
        """
        Build the context for the language model.
        """

        parts = []

        for paper in sorted(grouped_chunks):

            parts.append(f"# {paper}")

            for chunk in grouped_chunks[paper]:

                parts.append(chunk["text"])

            parts.append("")

        return "\n\n".join(parts)

    def build_prompt(self, context):
        """
        Build the LLM prompt.
        """

        return f"""
You are an expert academic researcher.

Below are the conclusion sections from several research papers.

For EACH paper, identify:

- Main limitations
- Remaining challenges
- Future work suggested by the authors

Then create a final section called:

# Common Research Gaps

where you summarize:

- Common limitations across the papers
- Open research problems
- Promising future research directions

Write the output in clean Markdown.

Context:

{context}
"""

    def save_markdown(self, text):
        """
        Save markdown report.
        """

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
        """
        Execute the pipeline.
        """

        grouped_chunks = self.collect_conclusions()

        context = self.build_context(
            grouped_chunks
        )

        prompt = self.build_prompt(
            context
        )

        report = self.llm.chat(prompt)

        self.save_markdown(report)

        return {
            "answer": (
                "Research gap analysis completed successfully.\n\n"
                "Generated file:\n"
                "- data/outputs/research_gap.md"
            ),
            "sources": [],
        }


if __name__ == "__main__":

    tool = ResearchGapFinder()

    result = tool.run()

    print(result["answer"])