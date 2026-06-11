import os
import tempfile
import streamlit as st
import nest_asyncio

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)

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
from llama_index.core.selectors import PydanticSingleSelector


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

st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 SmartRoute AI ")
st.markdown("Upload multiple files and ask questions .")


# -----------------------------
# FILE UPLOAD
# -----------------------------
with st.sidebar:

    uploaded_files = st.file_uploader(
        "Upload Files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    st.caption("Supported file formats: PDF, DOCX")
# -----------------------------
# LOAD & PROCESS PDFs
# -----------------------------
@st.cache_resource
def create_query_engine(_uploaded_files):

    all_documents = []

    for uploaded_file in _uploaded_files:

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        # Create temp file with original extension
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.getvalue()
            )

            temp_file_path = tmp_file.name

        # Select loader based on file type
        if file_extension == ".pdf":

            loader = PyPDFLoader(
                temp_file_path
            )

        elif file_extension == ".docx":

            loader = Docx2txtLoader(
                temp_file_path
            )

        elif file_extension in [".xlsx", ".xls"]:

            loader = UnstructuredExcelLoader(
                temp_file_path,
                mode="elements"
            )

        else:

            continue

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

            llama_docs.append(
                llama_doc
            )

        all_documents.extend(
            llama_docs
        )

        # Delete temp file
        os.unlink(
            temp_file_path
        )

    # Chunking
    splitter = SentenceSplitter(
        chunk_size=1500,
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
    query_engine=vector_index.as_query_engine(
        similarity_top_k=3
    ),
    description=(
        "Useful for retrieving specific facts, "
        "technical details, values, and references."
    )
)

    # Router Query Engine
    query_engine = RouterQueryEngine.from_defaults(
        selector=PydanticSingleSelector.from_defaults(),
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
        f"{len(uploaded_files)} file(s) uploaded successfully!"
    )

    with st.spinner("Processing files..."):

        query_engine = create_query_engine(
            uploaded_files
        )

    st.success("Documents indexed successfully!")

    # FILE COMPARISON SECTION
    file_names = [file.name for file in uploaded_files]

    st.markdown("### Compare Files")

    col1, col2 = st.columns(2)

    with col1:
        file1 = st.selectbox(
            "Select First File",
            file_names
        )

    with col2:
        file2 = st.selectbox(
            "Select Second File",
            file_names
        )

    compare_clicked = st.button(
        "Compare Files"
    )

    if compare_clicked:

        if file1 == file2:

            st.warning(
                "Please select two different files."
            )

        else:

            compare_prompt = f"""
            Compare document '{file1}' and document '{file2}'.

            Give:
        
            1. 5 Differences
            2. 2 Similarities
            3. Short Conclusion

            Keep the answer under 150 words.
            """

            with st.spinner(
                "Comparing files..."
            ):

                response = query_engine.query(
                    compare_prompt
                )

            st.markdown(
                "## Comparison Result"
            )

            st.write(
                response.response
            )

    # QUERY BOX
    query = st.text_input(
        "Ask a question about the files:"
    )

    if query:

        with st.spinner("Analyzing..."):

            response = query_engine.query(query)

        st.markdown("## Answer")

        st.write(response.response)

else:

    st.info("Please upload files to begin.")
