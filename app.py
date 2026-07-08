"""
Streamlit UI for RAG-Based Academic Research Assistant.

Place this file in the project root, next to:
- src/
- tools/
- data/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
TOOLS_DIR = PROJECT_ROOT / "tools"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="🎓",
    layout="wide",
)


# ------------------------------------------------------------
# Output file mapping
# ------------------------------------------------------------

TOOL_OUTPUTS: Dict[str, List[str]] = {
    "extract_experimental_results": [
        "results.csv",
    ],
    "compare_results": [
        "comparison_table.csv",
        "comparison_chart.png",
        "comparison_analysis.md",
    ],
    "generate_survey": [
        "mini_survey.md",
    ],
    "find_research_gap": [
        "research_gap.md",
    ],
    "compare_references": [
        "reference_year_distribution.csv",
        "reference_year_distribution.png",
        "reference_comparison.md",
    ],
    "compare_limitations": [
        "limitations_comparison.md",
    ],
}

PAGE_OUTPUTS: Dict[str, List[str]] = {
    "Experimental Results": [
        "results.csv",
    ],
    "Comparison": [
        "comparison_table.csv",
        "comparison_chart.png",
        "comparison_analysis.md",
    ],
    "Mini Survey": [
        "mini_survey.md",
    ],
    "Research Gap": [
        "research_gap.md",
    ],
    "References": [
        "reference_year_distribution.csv",
        "reference_year_distribution.png",
        "reference_comparison.md",
    ],
    "Limitations": [
        "limitations_comparison.md",
    ],
}


# ------------------------------------------------------------
# Lazy Agent loader
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_agent():
    """
    Load Agent only when it is actually needed.

    Important:
    The import is inside this function so Streamlit does not load Agent,
    Ollama, ChromaDB, or embedding models at app startup.
    """
    from src.agent import Agent

    return Agent()


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def output_path(filename: str) -> Path:
    return OUTPUT_DIR / filename


def get_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return "text/csv"

    if suffix == ".md":
        return "text/markdown"

    if suffix == ".png":
        return "image/png"

    return "application/octet-stream"


def show_download_button(path: Path, label: Optional[str] = None) -> None:
    if not path.exists():
        return

    counter_key = "_download_button_counter"

    if counter_key not in st.session_state:
        st.session_state[counter_key] = 0

    st.session_state[counter_key] += 1

    unique_key = f"download_{path.name}_{st.session_state[counter_key]}"

    st.download_button(
        label=label or f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime=get_mime_type(path),
        key=unique_key,
    )

def show_csv(path: Path) -> None:
    try:
        dataframe = pd.read_csv(path)
    except Exception as error:
        st.error(f"Could not read CSV file: {path.name}\n\n{error}")
        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )

    show_download_button(path)


def show_markdown_file(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as error:
        st.error(f"Could not read Markdown file: {path.name}\n\n{error}")
        return

    if not content.strip():
        st.warning(f"{path.name} exists, but it is empty.")
        return

    st.markdown(content)
    show_download_button(path)


def show_image(path: Path) -> None:
    st.image(
        str(path),
        width="stretch",
    )

    show_download_button(path)


def show_output_file(filename: str) -> None:
    path = output_path(filename)

    st.subheader(filename)

    if not path.exists():
        st.warning(f"{filename} was not found in: {OUTPUT_DIR}")
        return

    suffix = path.suffix.lower()

    if suffix == ".csv":
        show_csv(path)
    elif suffix == ".md":
        show_markdown_file(path)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        show_image(path)
    else:
        st.info(f"File exists, but preview is not supported: {filename}")
        show_download_button(path)


def show_output_files(filenames: Iterable[str]) -> None:
    for filename in filenames:
        show_output_file(filename)
        st.divider()


def format_pages(pages: str) -> str:
    if not pages:
        return ""

    if "-" not in pages:
        return f"Page {pages}"

    start, end = pages.split("-", maxsplit=1)

    if start == end:
        return f"Page {start}"

    return f"Pages {start}-{end}"


def show_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    st.markdown("#### Sources")

    for source in sources:
        paper = source.get("paper", "Unknown paper")
        section = str(source.get("section", "Unknown section")).title()
        pages = format_pages(str(source.get("pages", "")))

        st.markdown(f"- **{paper}** | {section} | {pages}")


def infer_tool_from_answer(result: Dict[str, Any]) -> str:
    """
    Some older tool outputs do not include the 'tool' key.
    This function keeps the UI stable even before agent.py is patched.
    """
    tool = result.get("tool")

    if tool:
        return str(tool)

    answer = str(result.get("answer", "")).lower()

    if "research gap" in answer:
        return "find_research_gap"

    if "reference comparison" in answer:
        return "compare_references"

    if "limitations comparison" in answer:
        return "compare_limitations"

    if "mini survey" in answer:
        return "generate_survey"

    if "comparison completed" in answer:
        return "compare_results"

    if "experimental results extracted" in answer:
        return "extract_experimental_results"

    return "rag"


def show_tool_outputs(tool_name: str, answer: str) -> None:
    files = TOOL_OUTPUTS.get(tool_name, [])

    if not files:
        return

    if "failed" in answer.lower():
        st.warning("The tool reported a failure, so generated files may be missing or outdated.")

    with st.expander("Generated / related output files", expanded=True):
        show_output_files(files)


def normalize_result(raw_result: Any) -> Dict[str, Any]:
    if isinstance(raw_result, dict):
        return raw_result

    return {
        "tool": "unknown",
        "answer": str(raw_result),
        "sources": [],
    }


# ------------------------------------------------------------
# UI sections
# ------------------------------------------------------------

def render_home_header() -> None:
    st.title("🎓 RAG-Based Academic Research Assistant")
    st.caption(
        "Streamlit UI for asking questions, running tools, and viewing generated CSV, PNG, and Markdown outputs."
    )


def render_chat_page() -> None:
    render_home_header()

    st.markdown("## Chat / Ask Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_area(
        "Write your question",
        placeholder="Example: Which paper achieved the highest AP?",
        height=120,
    )

    ask_clicked = st.button(
        "Ask",
        type="primary",
        width="stretch",
    )

    if ask_clicked:
        clean_question = question.strip()

        if not clean_question:
            st.warning("Please write a question first.")
        else:
            with st.spinner("Loading Agent and generating answer..."):
                try:
                    agent = get_agent()
                    raw_result = agent.answer(clean_question)
                    result = normalize_result(raw_result)
                except Exception as error:
                    st.error(f"Agent error:\n\n{error}")
                    result = None

            if result:
                st.session_state.chat_history.append(
                    {
                        "question": clean_question,
                        "result": result,
                    }
                )

    if not st.session_state.chat_history:
        st.info("Ask a question to see the answer here.")
        return

    st.markdown("## Conversation")

    for item in reversed(st.session_state.chat_history):
        user_question = item["question"]
        result = item["result"]

        tool_name = infer_tool_from_answer(result)
        answer = str(result.get("answer", "No answer returned."))
        sources = result.get("sources", [])

        with st.container(border=True):
            st.markdown("#### Question")
            st.markdown(user_question)

            st.markdown("#### Executed Tool")
            st.code(tool_name)

            if result.get("used_fallback"):
                st.warning("RAG did not find enough evidence, so the fallback LLM answer was used.")

            st.markdown("#### Answer")
            st.markdown(answer)

            show_sources(sources)
            show_tool_outputs(tool_name, answer)


def render_static_output_page(page_name: str) -> None:
    render_home_header()
    st.markdown(f"## {page_name}")

    st.caption(f"Reading files from: `{OUTPUT_DIR}`")
    show_output_files(PAGE_OUTPUTS[page_name])


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------

pages = [
    "Chat / Ask Question",
    "Experimental Results",
    "Comparison",
    "Mini Survey",
    "Research Gap",
    "References",
    "Limitations",
]

selected_page = st.sidebar.radio(
    "Navigation",
    pages,
)

st.sidebar.divider()
st.sidebar.markdown("### Output folder")
st.sidebar.code(str(OUTPUT_DIR))

if selected_page == "Chat / Ask Question":
    render_chat_page()
else:
    render_static_output_page(selected_page)
