import os
import streamlit as st
import nest_asyncio
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    SummaryIndex, 
    Settings, 
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

nest_asyncio.apply()

# 1. Setup Models (Cached)
@st.cache_resource
def init_models():
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    Settings.llm = Groq(model="llama-3.1-8b-instant")
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

init_models()

st.title("🤖 SmartRoute AI - RAG + Summarization") 
st.markdown("Ask questions from your uploaded PDF using RouteLLM + RAG")

# Sidebar for file upload
with st.sidebar:
    # PDF Upload Block
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

documents = []

if uploaded_files:
    for uploaded_file in uploaded_files:

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Add source file name
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        documents.extend(docs)

    st.sidebar.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
    # 2. Process Document (Cached)
    @st.cache_resource
    def create_query_engine(_file_path):
        reader = SimpleDirectoryReader(input_files=[_file_path])
        docs = reader.load_data()
        
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(docs)
        
        vector_index = VectorStoreIndex(nodes)
        summary_index = SummaryIndex(nodes)
        
        # Define Tools
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
            description="Useful for summarization and broad overview questions."
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_index.as_query_engine(),
            description="Useful for retrieving specific facts, numbers, or technical details."
        )
        
        return RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool],
            verbose=True
        )

    query_engine = create_query_engine(temp_path)

    # 3. Chat Interface
    query = st.text_input("Ask a question about the paper:")
    
    if query:
        with st.spinner("Analyzing..."):
            response = query_engine.query(query)
            st.markdown("### Answer")
            st.write(response.response)
else:
    st.info("Please upload a PDF to get started.")
