from pathlib import Path
import sys
import uuid

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "data" / "outputs"
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Academic RAG",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(0,229,255,.18), transparent 28%),
        radial-gradient(circle at top right, rgba(168,85,247,.15), transparent 30%),
        #050712;
    color: #eef7ff;
}
[data-testid="stSidebar"] {
    background: #080d1c;
    border-right: 1px solid rgba(0,229,255,.18);
}
.block-container { padding-top: 2rem; max-width: 1250px; }
h1, h2, h3, p, label, span { color: #eef7ff !important; }
.card {
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(0,229,255,.18);
    border-radius: 22px;
    padding: 20px;
    margin: 14px 0;
    box-shadow: 0 0 32px rgba(0,229,255,.06);
}
.hero {
    background:
        linear-gradient(135deg, rgba(15,23,42,.92), rgba(10,12,24,.82)),
        radial-gradient(circle at 15% 0%, rgba(0,229,255,.22), transparent 35%),
        radial-gradient(circle at 85% 20%, rgba(168,85,247,.18), transparent 35%);
    border: 1px solid rgba(0,229,255,.22);
    border-radius: 28px;
    padding: 30px;
    margin-bottom: 22px;
    box-shadow: 0 0 42px rgba(0,229,255,.08);
}
.hero h1 {
    font-size: 3rem;
    line-height: 1;
    margin: 0;
}
.grad {
    background: linear-gradient(90deg, #00e5ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.muted { color: #8ea4bf !important; }
.pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(0,229,255,.10);
    border: 1px solid rgba(0,229,255,.25);
    color: #baf7ff !important;
    font-weight: 700;
}
.stButton button, .stDownloadButton button {
    border-radius: 999px !important;
    border: 1px solid rgba(0,229,255,.35) !important;
    background: linear-gradient(135deg, rgba(0,229,255,.20), rgba(168,85,247,.20)) !important;
    color: white !important;
    font-weight: 700 !important;
}
textarea, input {
    background: rgba(8,13,28,.95) !important;
    color: white !important;
    border: 1px solid rgba(0,229,255,.22) !important;
    border-radius: 16px !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,229,255,.16);
    border-radius: 16px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_agent():
    from src.agent import Agent
    return Agent()


def path(name: str) -> Path:
    return OUTPUTS / name


def mime(p: Path) -> str:
    if p.suffix == ".csv":
        return "text/csv"
    if p.suffix == ".md":
        return "text/markdown"
    if p.suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def download(p: Path):
    st.download_button(
        "Download",
        data=p.read_bytes(),
        file_name=p.name,
        mime=mime(p),
        key=f"dl_{p.name}_{uuid.uuid4().hex}",
        width="stretch",
    )


def show_file(name: str):
    p = path(name)

    if not p.exists():
        st.warning(f"{name} not found in data/outputs.")
        return

    st.markdown(f'<div class="card"><h3>{name}</h3>', unsafe_allow_html=True)

    if p.suffix == ".csv":
        st.dataframe(pd.read_csv(p), width="stretch", hide_index=True)
    elif p.suffix == ".png":
        st.image(str(p), width="stretch")
    elif p.suffix == ".md":
        st.markdown(p.read_text(encoding="utf-8", errors="replace"))
    else:
        st.info("Preview is not available.")

    download(p)
    st.markdown("</div>", unsafe_allow_html=True)


def files_for_tool(tool: str):
    return {
        "extract_experimental_results": ["results.csv"],
        "compare_results": ["comparison_table.csv", "comparison_chart.png", "comparison_analysis.md"],
        "generate_survey": ["mini_survey.md"],
        "find_research_gap": ["research_gap.md"],
        "compare_references": ["reference_year_distribution.csv", "reference_year_distribution.png", "reference_comparison.md"],
        "compare_limitations": ["limitations_comparison.md"],
    }.get(tool, [])


def show_sources(sources):
    if not sources:
        return

    rows = []
    for s in sources:
        rows.append({
            "Paper": s.get("paper", ""),
            "Section": str(s.get("section", "")).title(),
            "Pages": s.get("pages", ""),
        })

    st.markdown('<div class="card"><h3>Sources</h3>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def chat_page():
    st.markdown("""
    <div class="hero">
        <span class="pill">Agent Powered · Neon Dark UI</span>
        <h1>RAG-Based <span class="grad">Academic Assistant</span></h1>
        <p class="muted">Ask research questions, generate academic outputs, and review CSV, PNG, and Markdown results in one dark workspace.</p>
    </div>
    """, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    q = st.text_area("Write your question", height=120)

    if st.button("Ask Agent", width="stretch", type="primary"):
        if not q.strip():
            st.warning("Write a question first.")
        else:
            try:
                with st.spinner("Agent is working..."):
                    result = load_agent().answer(q.strip())

                st.session_state.history.insert(0, {
                    "question": q.strip(),
                    "result": result,
                })

            except Exception as e:
                st.error(f"Agent error: {e}")

    for item in st.session_state.history:
        result = item["result"] if isinstance(item["result"], dict) else {}
        tool = result.get("tool", "unknown")
        answer = result.get("answer", str(item["result"]))
        sources = result.get("sources", [])

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Question")
        st.write(item["question"])
        st.markdown("#### Executed Tool")
        st.markdown(f'<span class="pill">{tool}</span>', unsafe_allow_html=True)
        st.markdown("#### Answer")
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        show_sources(sources)

        for f in files_for_tool(tool):
            show_file(f)


def output_page(title: str, files: list[str]):
    st.title(title)
    for f in files:
        show_file(f)


with st.sidebar:
    st.markdown("## ⚡ Academic RAG")
    st.caption("Research assistant dashboard")
    page = st.radio(
        "Navigation",
        [
            "Chat",
            "Experimental Results",
            "Comparison",
            "Mini Survey",
            "Research Gap",
            "References",
            "Limitations",
        ],
    )
    st.caption(str(OUTPUTS))

if page == "Chat":
    chat_page()
elif page == "Experimental Results":
    output_page("Experimental Results", ["results.csv"])
elif page == "Comparison":
    output_page("Comparison", ["comparison_table.csv", "comparison_chart.png", "comparison_analysis.md"])
elif page == "Mini Survey":
    output_page("Mini Survey", ["mini_survey.md"])
elif page == "Research Gap":
    output_page("Research Gap", ["research_gap.md"])
elif page == "References":
    output_page("References", ["reference_year_distribution.csv", "reference_year_distribution.png", "reference_comparison.md"])
elif page == "Limitations":
    output_page("Limitations", ["limitations_comparison.md"])
