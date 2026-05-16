import os
import tempfile
import streamlit as st
import nest_asyncio

from langchain_community.document_loaders import PyPDFLoader

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    Settings,
    Document
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


# Fix async issues
nest_asyncio.apply()


# -----------------------------
# MODEL SETUP
# -----------------------------
@st.cache_resource
def init_models():

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    Settings.llm = Groq(
        model="llama-3.1-8b-instant"
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


init_models()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="SmartRoute AI",
    layout="wide"
)

st.title("🤖 SmartRoute AI - Multi PDF RAG")
st.markdown("Upload multiple PDFs and ask questions using Router-based RAG.")


# -----------------------------
# FILE UPLOAD
# -----------------------------
with st.sidebar:

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True
    )


# -----------------------------
# LOAD & PROCESS PDFs
# -----------------------------
@st.cache_resource
def create_query_engine(_uploaded_files):

    all_documents = []

    for uploaded_file in _uploaded_files:

        # Create temp file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.getvalue())

            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)

        docs = loader.load()

        # Convert LangChain docs -> LlamaIndex docs
        llama_docs = []

        for doc in docs:

            llama_doc = Document(
                text=doc.page_content,
                metadata={
                    "source": uploaded_file.name,
                    **doc.metadata
                }
            )

            llama_docs.append(llama_doc)

        all_documents.extend(llama_docs)

        # Delete temp file
        os.unlink(temp_pdf_path)

    # Chunking
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )

    nodes = splitter.get_nodes_from_documents(
        all_documents
    )

    # Vector Index
    vector_index = VectorStoreIndex(nodes)

    # Summary Index
    summary_index = SummaryIndex(nodes)

    # Summary Tool
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize"
        ),
        description=(
            "Useful for summarization, broad overviews, "
            "and high-level understanding."
        )
    )

    # Vector Tool
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(),
        description=(
            "Useful for retrieving specific facts, "
            "technical details, values, and references."
        )
    )

    # Router Query Engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool
        ],
        verbose=True
    )

    return query_engine


# -----------------------------
# MAIN APP
# -----------------------------
if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully!"
    )

    with st.spinner("Processing PDFs..."):

        query_engine = create_query_engine(
            uploaded_files
        )

    st.success("Documents indexed successfully!")

    # Query Box
    query = st.text_input(
        "Ask a question about the PDFs:"
    )

    if query:

        with st.spinner("Analyzing..."):

            response = query_engine.query(query)

        st.markdown("## Answer")

        st.write(response.response)

else:

    st.info("Please upload PDF files to begin.")
