import streamlit as st
import time
import tempfile

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.schema import Document

st.set_page_config(page_title="RAG Document Q&A — Ollama", page_icon="🦙")
st.title("RAG Document Q&A — Ollama (Open Source)")

# ----------------------------
# Sidebar (no persistence; asked every refresh)
# ----------------------------
with st.sidebar:
    st.subheader("🛠️ Ollama Settings")
    base_url = st.text_input(
        "Ollama Base URL",
        value="http://localhost:11434",
        help="Local default: http://localhost:11434. If using a remote/hosted Ollama, put its URL here."
    )
    chat_model = st.text_input(
        "Chat Model",
        value="llama3.1",
        help="Examples: llama3.1, llama3, qwen2.5, mistral, phi4, etc."
    )
    embed_model = st.text_input(
        "Embedding Model",
        value="nomic-embed-text",
        help="Use an embedding-capable model available in your Ollama instance."
    )

    st.markdown("---")
    if st.session_state.get("vectors_ready"):
        st.success("✅ Vector Database is ready")
    else:
        st.info("ℹ️ Vector DB will be built on your first query")

    # ------- NEW: Built-in help on getting the "API" for free -------
    with st.expander("🆓 How to get the Ollama API for free (local setup)", expanded=False):
        st.markdown(
            """
**Fastest path (no API key needed): run Ollama locally.**

1. **Install Ollama** (macOS/Linux/Windows): https://ollama.com/download  
2. **Pull a chat model** (example):
   ```bash
   ollama pull llama3.1
