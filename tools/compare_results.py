"""
Compare the extracted experimental results from all papers.

Outputs:
    data/outputs/comparison_table.csv
    data/outputs/comparison_chart.png
    data/outputs/comparison_analysis.md
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.llm_service import LLMService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
)

RESULTS_PATH = OUTPUT_DIR / "results.csv"

TABLE_PATH = OUTPUT_DIR / "comparison_table.csv"

CHART_PATH = OUTPUT_DIR / "comparison_chart.png"

ANALYSIS_PATH = OUTPUT_DIR / "comparison_analysis.md"


class ComparisonService:
    """
    Compare experimental results across papers.
    """

    def __init__(self):

        self.llm = LLMService()

    def load_results(self):
        """
        Load results.csv.
        """

        if not RESULTS_PATH.exists():
            raise FileNotFoundError(
                "results.csv was not found."
            )

        return pd.read_csv(RESULTS_PATH)

    def build_table(self, dataframe):
        """
        Build a clean comparison table.
        """

        columns = [
            "paper",
            "model",
            "dataset",
            "backbone",
            "AP",
            "AP50",
            "AP75",
            "source_page",
        ]

        table = dataframe[columns].copy()

        return table

    def save_table(self, table):
        """
        Save comparison table.
        """

        table.to_csv(
            TABLE_PATH,
            index=False,
        )

        print(
            f"Comparison table saved:\n{TABLE_PATH}"
        )

    def get_best_paper(self, dataframe):
        """
        Return the paper with the highest AP.
        """

        dataframe = dataframe.copy()

        dataframe["AP"] = pd.to_numeric(
            dataframe["AP"],
            errors="coerce",
        )

        return dataframe.loc[
            dataframe["AP"].idxmax()
        ]

    def create_chart(self, dataframe):
        """
        Create AP comparison chart.
        """

        dataframe = dataframe.copy()

        dataframe["AP"] = pd.to_numeric(
            dataframe["AP"],
            errors="coerce",
        )

        plt.figure(figsize=(8, 5))

        plt.bar(
            dataframe["paper"],
            dataframe["AP"],
        )

        plt.title(
            "Average Precision Comparison"
        )

        plt.xlabel("Paper")

        plt.ylabel("AP")

        plt.tight_layout()

        plt.savefig(
            CHART_PATH,
            dpi=300,
        )

        plt.close()

        print(
            f"Chart saved:\n{CHART_PATH}"
        )

    def build_prompt(
        self,
        comparison_table,
        best_paper,
    ):
        """
        Build LLM prompt.
        """

        table_text = comparison_table.to_string(
            index=False
        )

        return f"""
You are an academic research assistant.

Below is a comparison of four object detection papers.

{table_text}

The paper with the highest AP is:

{best_paper["paper"]}

Write a short academic comparison.

Include:

- Which paper performed best.
- Compare the reported AP values.
- Compare the datasets.
- Compare the backbones.
- Mention important observations.
- Write a final conclusion.

Maximum length:
250 words.

Do not use markdown tables.
"""

    def generate_analysis(
        self,
        comparison_table,
        best_paper,
    ):
        """
        Generate comparison analysis using Qwen.
        """

        prompt = self.build_prompt(
            comparison_table,
            best_paper,
        )

        return self.llm.chat(prompt)

    def save_analysis(
        self,
        analysis,
    ):
        """
        Save generated analysis.
        """

        with open(
            ANALYSIS_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(analysis)

        print(
            f"Analysis saved:\n{ANALYSIS_PATH}"
        )

    def run(self):
        """
        Run the comparison pipeline.
        """

        dataframe = self.load_results()

        table = self.build_table(
            dataframe
        )

        self.save_table(
            table
        )

        self.create_chart(
            dataframe
        )

        best_paper = self.get_best_paper(
            dataframe
        )

        print()

        print(
            f"Best paper: {best_paper['paper']}"
        )

        print(
            f"Highest AP: {best_paper['AP']}"
        )

        analysis = self.generate_analysis(
            table,
            best_paper,
        )

        self.save_analysis(
            analysis
        )

        print()

        print("Comparison completed successfully.")


def main():

    service = ComparisonService()

    service.run()


if __name__ == "__main__":
    main()