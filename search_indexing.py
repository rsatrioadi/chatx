from typing import List
from langchain import FAISS
from constants import chunk_size, chunk_overlap, number_snippets_to_retrieve
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
import streamlit as st
import os
import tempfile

FAISS_INDEX_PATH = "faiss_index"


def download_and_index_file(files: List[st.runtime.uploaded_file_mgr.UploadedFile]) -> FAISS:
    """
    Process and index a list of uploaded files, then save the index to disk.

    Args:
        files (List[st.uploaded_file_manager.UploadedFile]): List of uploaded files.

    Returns:
        FAISS: The FAISS index created from the processed files.
    """
    all_pages = []

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        try:
            loader = UnstructuredFileLoader(temp_file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            pages = loader.load_and_split(splitter)
            pages = filter_complex_metadata(pages)

            for i, page in enumerate(pages):
                page.metadata['page'] = i
                page.metadata['url'] = uploaded_file.name
                page.metadata['title'] = uploaded_file.name

            all_pages.extend(pages)
        finally:
            os.remove(temp_file_path)

    faiss_index = FAISS.from_documents(all_pages, SentenceTransformerEmbeddings())
    faiss_index.save_local(FAISS_INDEX_PATH)
    return faiss_index


def load_faiss_index() -> FAISS:
    """
    Load the FAISS index from disk.

    Returns:
        FAISS: The loaded FAISS index.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, SentenceTransformerEmbeddings())
    return None


def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    """
    Search a FAISS index, using the passed query
    """

    docs = faiss_index.similarity_search(query, k=top_k)

    return docs
