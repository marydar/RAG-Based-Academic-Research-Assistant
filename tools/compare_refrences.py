"""
compare_references.py

Analyze the references of all papers.

Part 1:
- Read references
- Extract publication years
- Build statistics
- Save CSV
- Generate chart
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.vector_store import load_chunks
from src.llm_service import LLMService


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
)

CSV_PATH = OUTPUT_DIR / "reference_year_distribution.csv"

CHART_PATH = OUTPUT_DIR / "reference_year_distribution.png"


class ReferenceComparer:

    TARGET_SECTION = "references"

    YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

    def __init__(self):

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.llm = LLMService()

    # --------------------------------------------------
    # Load reference sections
    # --------------------------------------------------

    def collect_references(self):

        chunks = load_chunks()

        grouped = {}

        for chunk in chunks:

            section = (
                chunk.get("section", "")
                .lower()
                .strip()
            )

            if section != self.TARGET_SECTION:
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
                "No reference sections found."
            )

        return grouped

    # --------------------------------------------------
    # Extract years
    # --------------------------------------------------

    def extract_years(self, grouped):

        statistics = {}
        all_years = set()

        for paper, texts in grouped.items():

            years = []

            for text in texts:

                matches = self.YEAR_PATTERN.findall(text)

                for year in matches:

                    year = int(year)

                    # Ignore obvious OCR mistakes
                    if 1950 <= year <= 2035:
                        years.append(year)

            counter = Counter(years)

            statistics[paper] = {
                "years": years,
                "counts": counter,
                "total": len(years),
                "oldest": min(years) if years else None,
                "newest": max(years) if years else None,
                "average": (
                    round(sum(years) / len(years), 1)
                    if years else None
                ),
            }

            all_years.update(counter.keys())

        return statistics, sorted(all_years)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    def save_csv(
        self,
        statistics,
        years,
    ):

        rows = []

        papers = sorted(statistics.keys())

        for year in years:

            row = {
                "Year": year
            }

            for paper in papers:

                row[paper] = (
                    statistics[paper]["counts"]
                    .get(year, 0)
                )

            rows.append(row)

        dataframe = pd.DataFrame(rows)

        dataframe.to_csv(
            CSV_PATH,
            index=False,
        )

        return dataframe

    # --------------------------------------------------
    # Generate chart
    # --------------------------------------------------

    def generate_chart(
        self,
        dataframe,
    ):

        plt.figure(figsize=(12, 6))

        for column in dataframe.columns[1:]:

            plt.plot(
                dataframe["Year"],
                dataframe[column],
                marker="o",
                label=column,
            )

        plt.title(
            "Reference Publication Year Distribution"
        )

        plt.xlabel(
            "Publication Year"
        )

        plt.ylabel(
            "Number of References"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            CHART_PATH,
            dpi=300,
        )

        plt.close()
        
        # --------------------------------------------------
    # Build LLM context
    # --------------------------------------------------
    
    def dataframe_to_markdown(
        self,
        dataframe,
    ):
        """
        Convert the year distribution dataframe to a markdown table.
        """

        return dataframe.to_markdown(index=False)

    def build_context(
        self,
        statistics,
    ):
        """
        Convert the reference statistics into a compact context
        for the language model.
        """

        context = []

        for paper in sorted(statistics.keys()):

            info = statistics[paper]

            context.append(f"# {paper}")

            context.append(
                f"Total references: {info['total']}"
            )

            context.append(
                f"Oldest reference: {info['oldest']}"
            )

            context.append(
                f"Newest reference: {info['newest']}"
            )

            context.append(
                f"Average publication year: {info['average']}"
            )

            context.append("\nPublication year distribution:")

            for year in sorted(info["counts"]):

                context.append(
                    f"- {year}: {info['counts'][year]}"
                )

            context.append("")

        return "\n".join(context)


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    def build_prompt(
        self,
        context,
    ):
        """
        Build the prompt for the language model.
        """

        return f"""
    You are an expert researcher.

    Below is the publication-year distribution of the references
    used by several academic papers.

    Write a professional Markdown report with the following sections.

    # Reference Statistics

    Create a markdown table with these columns:

    | Paper | Total References | Oldest | Newest | Average Year |

    Fill every row.

    ---

    # Historical Comparison

    Compare the papers historically.

    Discuss questions such as:

    - Which paper relies more on classic work?
    - Which paper cites the newest research?
    - Which paper appears to have the broadest literature review?
    - Which papers build upon previous DETR-family work?

    ---

    # Research Trend

    Describe how the cited literature evolves over time.

    Comment on:

    - older references
    - recent references
    - transition toward Transformer-based object detection

    ---

    # Observations

    Provide 5-8 concise observations.

    ---

    # Conclusion

    Summarize what the reference analysis reveals.

    Return ONLY clean Markdown.

    Statistics:

    {context}
    """


    # --------------------------------------------------
    # Save markdown
    # --------------------------------------------------

    def save_markdown(
        self,
        markdown,
    ):
        """
        Save the generated report.
        """

        markdown_path = (
            OUTPUT_DIR
            / "reference_comparison.md"
        )

        with markdown_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)

        return markdown_path
    
        # --------------------------------------------------
    # Execute tool
    # --------------------------------------------------

    def run(self):
        """
        Execute the complete reference analysis pipeline.
        """

        grouped = self.collect_references()

        statistics, years = self.extract_years(
            grouped
        )

        dataframe = self.save_csv(
            statistics,
            years,
        )
        

        self.generate_chart(
            dataframe,
        )
        
        year_table = self.dataframe_to_markdown(dataframe)

        context = self.build_context(
            statistics,
        )

        prompt = self.build_prompt(
            context,
        )

        report = self.llm.chat(
            prompt
        )
        clean_report = "\n".join(
            line.lstrip()
            for line in report.splitlines()
        )

        final_markdown = f"""# Publication Year Distribution

        {year_table}

        ---

        {clean_report}
        """

        self.save_markdown(
            final_markdown
        )

        return {
            "answer": (
                "Reference comparison completed successfully.\n\n"
                "Generated files:\n"
                "- data/outputs/reference_year_distribution.csv\n"
                "- data/outputs/reference_year_distribution.png\n"
                "- data/outputs/reference_comparison.md"
            ),
            "sources": [],
        }


if __name__ == "__main__":

    tool = ReferenceComparer()

    result = tool.run()

    print(result["answer"])