"""
Collect the important sections from every paper that will later
be used to generate a mini survey.
"""

from pathlib import Path

import chromadb
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_service import LLMService
from src.vector_store import (
    DB_PATH,
    COLLECTION_NAME,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
)

SURVEY_PATH = (
    OUTPUT_DIR
    / "mini_survey.md"
)

COMPARISON_TABLE_PATH = (
    OUTPUT_DIR
    / "comparison_table.csv"
)

TARGET_SECTIONS = {
    "abstract",
    "introduction",
    "methodology",
    "conclusion",
}


class SurveyGenerator:
    """
    Generate a mini literature survey.
    """

    def __init__(self):

        client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = client.get_collection(
            name=COLLECTION_NAME
        )

        self.llm = LLMService()

    def load_sections(self):
        """
        Load the required sections from every paper.
        """

        results = self.collection.get(
            where={
                "retrieval_enabled": True
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        grouped = {}

        for document, metadata in zip(
            results["documents"],
            results["metadatas"],
        ):

            section = (
                metadata.get(
                    "section",
                    "",
                )
                .lower()
                .strip()
            )

            if section not in TARGET_SECTIONS:
                continue

            paper = metadata.get(
                "paper_short_name",
                metadata.get(
                    "paper_title",
                    metadata.get(
                        "paper_id",
                        "Unknown",
                    ),
                ),
            )

            if paper not in grouped:
                grouped[paper] = []

            grouped[paper].append(
                {
                    "section": section,
                    "page_start": metadata.get(
                        "page_start"
                    ),
                    "page_end": metadata.get(
                        "page_end"
                    ),
                    "text": document,
                }
            )

        return grouped
    
    def build_summary_prompt(
        self,
        paper,
    ):
        """
        Build the prompt for summarizing one paper.
        """

        return f"""
    You are reading an academic paper.

    Write a concise research summary.

    Paper:
    {paper["paper"]}

    Abstract:
    {paper["abstract"]}

    Introduction:
    {paper["introduction"]}

    Methodology:
    {paper["methodology"]}

    Conclusion:
    {paper["conclusion"]}

    Requirements:

    - Maximum 150 words.
    - Explain the research problem.
    - Explain the proposed method.
    - Mention the main contribution.
    - Mention the main conclusion.
    - Use academic language.
    """
    
    
    def summarize_paper(
        self,
        paper,
    ):
        """
        Generate a summary for one paper.
        """

        prompt = self.build_summary_prompt(
            paper
        )

        return self.llm.chat(
            prompt
        )
        
    def generate_summaries(
        self,
        merged_papers,
    ):
        """
        Generate summaries for every paper.
        """

        summaries = {}

        for paper_name, paper in merged_papers.items():

            print(
                f"Summarizing {paper_name}..."
            )

            summary = self.summarize_paper(
                paper
            )

            summaries[paper_name] = summary

        return summaries
    
    def merge_sections(
            self,
            papers,
        ):
        """
        Merge all chunks that belong to the same section.
        """

        merged = {}

        for paper, chunks in papers.items():

            merged[paper] = {
                "paper": paper,
                "abstract": "",
                "introduction": "",
                "methodology": "",
                "conclusion": "",
            }

            chunks.sort(
                key=lambda chunk: (
                    chunk["page_start"],
                    chunk["section"],
                )
            )

            for chunk in chunks:

                section = chunk["section"]

                merged[paper][section] += (
                    chunk["text"]
                    + "\n\n"
                )

        return merged

    def print_sections(
        self,
        papers,
    ):
        """
        Print the collected sections.
        """

        for paper, chunks in papers.items():

            print()
            print("=" * 80)
            print(paper)
            print("=" * 80)

            chunks.sort(
                key=lambda chunk: (
                    chunk["page_start"],
                    chunk["section"],
                )
            )

            for chunk in chunks:

                print()

                print(
                    f"Section: {chunk['section']}"
                )

                print(
                    f"Pages: {chunk['page_start']} - {chunk['page_end']}"
                )

                print("-" * 60)

                print(chunk["text"][:500])

                print()
                
    def load_comparison_table(self):
        """
        Load the comparison table.
        """

        if not COMPARISON_TABLE_PATH.exists():
            raise FileNotFoundError(
                "comparison_table.csv not found."
            )

        return COMPARISON_TABLE_PATH.read_text(
            encoding="utf-8"
        )
        
    def build_survey_prompt(
        self,
        summaries,
        comparison_table,
    ):
        """
        Build the prompt for the final survey.
        """

        text = ""

        for paper, summary in summaries.items():

            text += f"""

    ==========================
    {paper}
    ==========================

    {summary}

    """

        return f"""
    You are an academic researcher.

    You have summaries of four research papers.

    {text}

    Below is the comparison table extracted from the papers.

    {comparison_table}

    Write a professional mini survey in Markdown.

    Structure:

    # Introduction

    Explain the object detection problem and why DETR-based methods are important.

    # Paper 1

    Summarize the first paper.

    # Paper 2

    Summarize the second paper.

    # Paper 3

    Summarize the third paper.

    # Paper 4

    Summarize the fourth paper.

    # Method Comparison

    Compare the proposed methods.

    Highlight the main differences.

    # Research Progress

    Explain how the methods evolved from one paper to another.

    # Experimental Comparison

    Use ONLY the comparison table.

    Discuss:

    - AP
    - AP50
    - AP75
    - backbone
    - datasets

    Mention which paper achieved the highest AP.

    # Conclusion

    Summarize the overall findings.

    Requirements:

    - Academic writing style.
    - Use Markdown headings.
    - Around 1000 words.
    - Do not invent information.
    - Base the survey only on the provided summaries and comparison table.
    """
    
    def generate_survey(
        self,
        summaries,
    ):
        """
        Generate the final survey.
        """

        comparison_table = self.load_comparison_table()

        prompt = self.build_survey_prompt(
            summaries,
            comparison_table,
        )

        return self.llm.chat(
            prompt
        )
        
    def save_survey(
        self,
        survey,
    ):
        """
        Save the generated survey.
        """

        with open(
            SURVEY_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(survey)

        print()

        # print(
        #     f"Survey saved to:\n{SURVEY_PATH}"
        # )

    def run(self):

        papers = self.load_sections()

        merged = self.merge_sections(
            papers
        )

        summaries = self.generate_summaries(
            merged
        )

        survey = self.generate_survey(
            summaries
        )

        self.save_survey(
            survey
        )

        print()

        print(
            "Mini survey generated successfully."
        )


def main():

    generator = SurveyGenerator()

    generator.run()


if __name__ == "__main__":
    main()