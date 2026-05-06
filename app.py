# !pip install llama-index==0.12.9
# !pip install llama-index-llms-groq
# !pip install llama-index-embeddings-huggingface
# !pip install sentence-transformers
# !pip install streamlit
import os
import streamlit as st
import nest_asyncio

nest_asyncio.apply()

# LlamaIndex Imports
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# ---------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------
st.set_page_config(
    page_title="SmartRoute AI - RAG Summarization",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SmartRoute AI - RAG + Summarization")
st.markdown("Ask questions from your uploaded PDF using RouteLLM + RAG")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Enter GROQ API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

# ---------------------------
# SAVE UPLOADED FILE
# ---------------------------
if uploaded_file is not None:

    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully ✅")

    # ---------------------------
    # LOAD DOCUMENTS
    # ---------------------------
    with st.spinner("Loading documents..."):
        documents = SimpleDirectoryReader(
            input_files=[uploaded_file.name]
        ).load_data()

    st.write(f"Number of documents loaded: {len(documents)}")

    # ---------------------------
    # CHUNKING
    # ---------------------------
    with st.spinner("Splitting documents into chunks..."):

        splitter = SentenceSplitter(
            chunk_size=2000,
            chunk_overlap=20
        )

        nodes = splitter.get_nodes_from_documents(documents)

    st.success(f"Created {len(nodes)} chunks")

    # ---------------------------
    # SET LLM + EMBEDDING MODEL
    # ---------------------------
    if api_key:

        os.environ["GROQ_API_KEY"] = api_key

        with st.spinner("Initializing models..."):

            Settings.llm = Groq(
                model="llama-3.1-8b-instant"
            )

            Settings.embed_model = HuggingFaceEmbedding()

        st.success("Models Initialized Successfully ✅")

        # ---------------------------
        # CREATE INDEXES
        # ---------------------------
        with st.spinner("Creating indexes..."):

            summary_index = SummaryIndex(nodes)
            vector_index = VectorStoreIndex(nodes)

        st.success("Indexes Created Successfully ✅")

        # ---------------------------
        # QUERY ENGINES
        # ---------------------------
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True
        )

        vector_query_engine = vector_index.as_query_engine()

        # ---------------------------
        # TOOLS
        # ---------------------------
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                "Useful for summarization questions related to the document"
            )
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context from the document"
            )
        )

        # ---------------------------
        # ROUTER QUERY ENGINE
        # ---------------------------
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool
            ]
        )

        st.success("Router Query Engine Ready 🚀")

        # ---------------------------
        # USER INPUT
        # ---------------------------
        st.subheader("Ask Questions")

        user_query = st.text_input(
            "Enter your question"
        )

        if st.button("Generate Response"):

            if user_query:

                with st.spinner("Generating response..."):
                    response = query_engine.query(user_query)

                st.subheader("Response")
                st.write(response)

            else:
                st.warning("Please enter a question")

    else:
        st.warning("Please enter GROQ API Key")
