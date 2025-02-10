"""Streamlit application for RAG-based document question answering.

This module provides a web interface for uploading documents and querying them
using RAG (Retrieval Augmented Generation) technology.
"""

import os
from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from config import config
from populate_database import DocumentProcessor
from query_data import RAGQueryEngine

# Page configuration
st.set_page_config(
    page_title="DocWhisperer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #ffffff;
    }
    .document-management {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .document-stats {
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        color: #333333;
    }
    .main-content {
        padding: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGQueryEngine()


def save_uploaded_files(uploaded_files: List[UploadedFile]) -> List[str]:
    """Save uploaded files to the data directory and return their paths."""
    saved_paths = []
    os.makedirs(config.DATA_PATH, exist_ok=True)

    for uploaded_file in uploaded_files:
        try:
            if not uploaded_file.name.lower().endswith(".pdf"):
                st.warning(
                    f"Skipping {uploaded_file.name}: Only PDF files are supported"
                )
                continue

            file_path = os.path.join(config.DATA_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {str(e)}")

    return saved_paths


def display_file_stats():
    """Display statistics about processed files."""
    try:
        files = os.listdir(config.DATA_PATH)
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        if pdf_files:
            st.markdown(
                f"""
                <div class="document-stats">
                    📊 <b>Current Status:</b> {len(pdf_files)} document(s) processed
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")


def main():
    """Main application entry point."""
    st.title("📚 DocWhisperer")
    st.markdown(
        """
        Upload PDF documents and ask questions about their contents.
        The assistant will provide detailed answers based on the document context.
        """
    )

    # Document Management Section
    st.markdown(
        """
        <div class="document-management">
        <h3>📁 Document Management</h3>
        """,
        unsafe_allow_html=True,
    )

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        accept_multiple_files=True,
        type=["pdf"],
        help="Select one or more PDF files to process",
    )

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_files:
            if st.button("📥 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    saved_paths = save_uploaded_files(uploaded_files)
                    if saved_paths:
                        st.session_state.processor.process_documents()
                        st.success(f"✅ Processed {len(saved_paths)} documents!")

    with col2:
        if st.button("🗑️ Reset Database", use_container_width=True, type="secondary"):
            with st.spinner("Resetting..."):
                st.session_state.processor.clear_database()
                # Add file cleanup
                for f in os.listdir(config.DATA_PATH):
                    os.remove(os.path.join(config.DATA_PATH, f))
                st.success("✅ Database reset complete!")

    display_file_stats()
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Interface
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Query input
    if prompt := st.chat_input(
        "Ask a question about the documents...", key="chat_input"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    response_text, sources, thinking = (
                        st.session_state.rag_engine.query(prompt)
                    )

                    # Main response
                    st.markdown("### 📝 Response")
                    st.markdown(response_text)

                    # Sources and analysis in expandable sections
                    with st.expander("📚 View Sources", expanded=False):
                        if sources:
                            for source in sources:
                                if source:
                                    parts = source.split(" ")[1].split(":")
                                    filename = os.path.basename(parts[0])
                                    page_info = f"Page {parts[1]}"
                                    chunk_info = f"Chunk {parts[-1]}"
                                    st.markdown(
                                        f"- **{filename}** ({page_info}, {chunk_info})"
                                    )
                        else:
                            st.info("No specific sources found for this response.")

                    with st.expander("🔍 View Analysis", expanded=False):
                        if thinking:
                            st.code(thinking, language="markdown")
                        else:
                            st.info("No detailed analysis available for this response.")

                    # Add to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info(
                        "Please try rephrasing your question or check if documents are properly loaded."
                    )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
