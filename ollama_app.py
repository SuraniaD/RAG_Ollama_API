import streamlit as st
import time
import tempfile
from pathlib import Path
import requests

# --- LLM + Embeddings ---
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# --- Text Splitters ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Chains ---
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Prompt Templates ---
from langchain_core.prompts import ChatPromptTemplate

# --- Vector Store ---
from langchain_community.vectorstores import FAISS

# --- Document Loaders ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader
)

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="RAG Document Q&A — Ollama", page_icon="🦙")
st.title("RAG Document Q&A — Ollama (Open Source)")

# ----------------------------
# Session State Init
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "vectors_ready" not in st.session_state:
    st.session_state["vectors_ready"] = False

# ----------------------------
# Sidebar (settings)
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

    if st.button("♻️ Reset Vector DB"):
        st.session_state["vectorstore"] = None
        st.session_state["vectors_ready"] = False
        st.success("Vector DB cleared. It will be rebuilt on the next question.")

# ----------------------------
# Helper: check Ollama is alive
# ----------------------------
def check_ollama(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ----------------------------
# Helper: Load documents from uploaded files
# ----------------------------
def load_documents(uploaded_files):
    """Convert Streamlit UploadedFile objects to LangChain Documents."""
    all_docs = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        # Save to a temporary file so the loaders can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            # Fallback for other formats (docx, etc.) – requires unstructured dependencies
            loader = UnstructuredFileLoader(tmp_path)

        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

# ----------------------------
# Main UI
# ----------------------------
st.header("1️⃣ Upload Documents")
uploaded_files = st.file_uploader(
    "Upload one or more files (PDF, TXT, MD, or other supported formats)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s).")
else:
    st.info("Upload some documents to ask questions about them.")

st.header("2️⃣ Ask Questions")
user_question = st.text_input("Ask a question about your documents:")

ask_button = st.button("Ask")

# ----------------------------
# Build Vector Store (first time on ask)
# ----------------------------
def build_vectorstore_if_needed():
    if st.session_state["vectorstore"] is not None:
        return  # Already built

    if not uploaded_files:
        st.error("Please upload at least one document before asking a question.")
        return

    if not check_ollama(base_url):
        st.error(
            f"Can't reach Ollama at {base_url}. "
            "Make sure Ollama is running and accessible."
        )
        return

    with st.spinner("📚 Loading documents and building vector store..."):
        # 1. Load docs
        docs = load_documents(uploaded_files)

        if not docs:
            st.error("No content could be loaded from the uploaded files.")
            return

        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Create embeddings and vector store
        embeddings = OllamaEmbeddings(
            base_url=base_url,
            model=embed_model,
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state["vectorstore"] = vectorstore
        st.session_state["vectors_ready"] = True

        st.success(f"✅ Vector store built with {len(chunks)} chunks.")

# ----------------------------
# RAG Q&A Logic
# ----------------------------
def answer_question(question: str):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    # Ensure vectorstore exists
    if st.session_state["vectorstore"] is None:
        build_vectorstore_if_needed()
        # If still None after trying to build, abort
        if st.session_state["vectorstore"] is None:
            return

    # Create retriever
    retriever = st.session_state["vectorstore"].as_retriever()

    # LLM
    llm = ChatOllama(
        base_url=base_url,
        model=chat_model,
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant. Use the following context from the documents
to answer the user's question. If the answer cannot be found in the context,
say you don't know instead of making something up.

<context>
{context}
</context>

Question: {input}

Answer in clear, concise language:
"""
    )

    # Create document chain and retrieval chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    with st.spinner("🤖 Thinking..."):
        result = rag_chain.invoke({"input": question})

    # Display answer
    st.markdown("### ✅ Answer")
    st.write(result["answer"])

    # Display retrieved context chunks
    with st.expander("🔍 Show retrieved context"):
        for i, doc in enumerate(result["context"]):
            st.markdown(f"**Chunk {i+1} — Source:** `{doc.metadata.get('source', 'unknown')}`")
            st.write(doc.page_content[:1200])  # truncate for display

# ----------------------------
# Trigger on button click
# ----------------------------
if ask_button and user_question:
    answer_question(user_question)
