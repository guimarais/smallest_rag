import tempfile
from pathlib import Path

import streamlit as st

from rag.config import load_config
from rag.models import check_ollama_running, model_exists, pull_model
from rag.pipeline import RAGPipeline

DATA_DIR = Path("data")
CONFIG_PATH = "config.yaml"

st.set_page_config(page_title="smallest_rag", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline() -> RAGPipeline:
    config = load_config(CONFIG_PATH)
    return RAGPipeline(config, DATA_DIR)


# ---------------------------------------------------------------------------
# Startup: ensure Ollama is running and models are available
# ---------------------------------------------------------------------------

def ensure_models_ready(pipeline: RAGPipeline) -> None:
    if st.session_state.get("models_ready"):
        return

    if not check_ollama_running():
        st.error(
            "Ollama is not running. Please start it with `ollama serve` and refresh."
        )
        st.stop()

    config = load_config(CONFIG_PATH)
    required_models = [config.llm_model, config.embedding_model]

    for model_name in required_models:
        if not model_exists(model_name):
            with st.status(f"Pulling model **{model_name}** …", expanded=True) as status:
                last_digest = None
                for progress in pull_model(model_name):
                    digest = getattr(progress, "digest", None)
                    total = getattr(progress, "total", None)
                    completed = getattr(progress, "completed", None)
                    detail = getattr(progress, "status", "")

                    if digest and digest != last_digest:
                        last_digest = digest

                    if total and completed:
                        pct = int(completed / total * 100)
                        status.update(
                            label=f"Pulling **{model_name}** … {pct}% ({detail})"
                        )
                    else:
                        status.update(label=f"Pulling **{model_name}** … {detail}")

                status.update(label=f"Model **{model_name}** ready.", state="complete")

    st.session_state["models_ready"] = True


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of {role, content, sources}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(pipeline: RAGPipeline) -> str:
    config = load_config(CONFIG_PATH)

    with st.sidebar:
        st.title("smallest_rag")

        st.subheader("Models")
        st.caption(f"LLM: `{config.llm_model}`")
        st.caption(f"Embeddings: `{config.embedding_model}`")

        st.divider()

        st.subheader("Search mode")
        search_mode = st.radio(
            "Mode",
            options=["hybrid", "cosine", "bm25"],
            index=["hybrid", "cosine", "bm25"].index(
                config.retrieval.search_mode
            ),
            label_visibility="collapsed",
        )

        st.divider()

        st.subheader("Knowledge base")
        sources = pipeline.list_sources()
        if sources:
            for src in sources:
                col1, col2 = st.columns([4, 1])
                col1.caption(src)
                if col2.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                    pipeline.delete_source(src)
                    st.rerun()
        else:
            st.caption("No documents ingested yet.")

        st.divider()

        st.subheader("Add document")
        uploaded = st.file_uploader(
            "Upload file",
            type=["txt", "pdf", "md", "docx"],
            label_visibility="collapsed",
        )
        if uploaded and st.button("Add to knowledge base"):
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix, delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            progress_bar = st.progress(0, text="Embedding chunks…")

            def update_progress(frac: float) -> None:
                progress_bar.progress(frac, text=f"Embedding chunks… {int(frac*100)}%")

            try:
                n = pipeline.ingest(tmp_path, uploaded.name, update_progress)
                progress_bar.empty()
                st.success(f"Added **{uploaded.name}** ({n} chunks).")
                st.rerun()
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Ingestion failed: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)

        st.divider()

        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()

    return search_mode


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

def render_chat(pipeline: RAGPipeline, search_mode: str) -> None:
    st.header("Chat")

    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(f"**{src['source']}**")
                        st.markdown(f"> {src['text'][:300]}…" if len(src["text"]) > 300 else f"> {src['text']}")

    question = st.chat_input("Ask a question…")
    if not question:
        return

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Build history for the model (exclude sources metadata)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["history"]
    ]

    # Retrieve context
    with st.spinner("Retrieving…"):
        chunks = pipeline.retrieve(question, search_mode)

    # Stream the assistant response
    with st.chat_message("assistant"):
        response_text = st.write_stream(
            pipeline.stream_response(question, chunks, history)
        )
        if chunks:
            with st.expander("Sources"):
                for src in chunks:
                    st.caption(f"**{src['source']}**")
                    text = src["text"]
                    st.markdown(f"> {text[:300]}…" if len(text) > 300 else f"> {text}")

    # Persist to history
    st.session_state["history"].append({"role": "user", "content": question, "sources": []})
    st.session_state["history"].append(
        {"role": "assistant", "content": response_text, "sources": chunks}
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_session()
    pipeline = load_pipeline()
    ensure_models_ready(pipeline)
    search_mode = render_sidebar(pipeline)
    render_chat(pipeline, search_mode)


main()
