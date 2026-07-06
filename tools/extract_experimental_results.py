"""
Collects all Experiments and Results sections from every paper.
"""
import sys
from pathlib import Path
import chromadb
import json
import csv
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import DB_PATH, COLLECTION_NAME
from src.llm_service import LLMService

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
)

CSV_PATH = OUTPUT_DIR / "results.csv"

TARGET_SECTIONS = {
    "experiments",
    "results",
    "experimental results",
    "evaluation",
    "evaluation results",
}

REQUIRED_FIELDS = [
    "paper",
    "model",
    "dataset",
    "backbone",
    "AP",
    "AP50",
    "AP75",
    "source_page",
]

class ExperimentalResultsExtractor:

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
        Load all chunks from the Experiments and Results sections.
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

            section = metadata.get("section", "").lower().strip()

            if section not in TARGET_SECTIONS:
                continue

            paper_id = metadata["paper_id"]

            if paper_id not in grouped:
                grouped[paper_id] = []

            grouped[paper_id].append(
                {
                    "text": document,
                    "metadata": metadata,
                }
            )

        return grouped

    def build_prompt(self, paper_id, chunks):
        """
        Build the extraction prompt for one paper.
        """
        paper_title = chunks[0]["metadata"]["paper_title"]
        paper_short_name = chunks[0]["metadata"]["paper_short_name"]
        context = []

        for chunk in chunks:

            meta = chunk["metadata"]

            context.append(
                f"""
    Section: {meta['section']}
    Pages: {meta['page_start']} - {meta['page_end']}

    {chunk['text']}
    """
            )

        context = "\n".join(context)

        return f"""
    You are extracting experimental results from an academic paper.

    Paper ID:
    {paper_id}
    

    Below are the Experiments and Results sections.

    {context}

    Extract ONE complete experiment from the paper.

    The values (model, backbone, AP, AP50, AP75, dataset) MUST come from the SAME table row.

    Do NOT combine information from different rows.

    Return the row with the highest AP.

    Return ONLY valid JSON.
    

    The paper name is already known.

    Paper:
    DAB-DETR

    Do NOT extract or modify the paper name.

    Return ONLY the following JSON:

    {{
        "model": null,
        "dataset": null,
        "backbone": null,
        "AP": null,
        "AP50": null,
        "AP75": null,
        "source_page": null
    }}
"""

    def extract_paper(self, paper_id, chunks):
        """
        Extract experimental results for one paper.
        """

        prompt = self.build_prompt(
            paper_id,
            chunks,
        )

        response = self.llm.chat(prompt)

        return response
    
    def parse_response(self, response):
        """
        Parse the JSON returned by Qwen.
        """

        response = response.strip()

        # Remove markdown code fences
        response = re.sub(
            r"^```(?:json)?",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(
            r"```$",
            "",
            response,
        )

        response = response.strip()

        # Extract first JSON object
        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No JSON object found."
            )

        response = response[start:end + 1]

        return json.loads(response)
    
    def validate_result(self, result):
        """
        Ensure every required field exists.
        """

        for field in REQUIRED_FIELDS:

            if field not in result:
                result[field] = None

        return result
    
    def save_csv(self, rows):
        """
        Save extracted results.
        """

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=REQUIRED_FIELDS,
            )

            writer.writeheader()

            writer.writerows(rows)

    # print(f"\nSaved to:\n{CSV_PATH}")
    
    def run(self):
        """
        Run the complete extraction pipeline.
        """

        papers = self.load_sections()

        rows = []

        for paper_id, chunks in papers.items():

            print(f"Extracting {paper_id}...")

            response = self.extract_paper(
                paper_id,
                chunks,
            )

            result = self.parse_response(
                response
            )
            result = {
                "paper": chunks[0]["metadata"]["paper_short_name"],
                **result,
            }
            result = self.validate_result(
                result
            )

            rows.append(result)

        self.save_csv(rows)

        return rows
if __name__ == "__main__":

    extractor = ExperimentalResultsExtractor()

    rows = extractor.run()

    print()

    for row in rows:
        print(row)