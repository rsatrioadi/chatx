from langchain import FAISS
from constants import chunk_size, chunk_overlap, number_snippets_to_retrieve
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import tempfile


def download_and_index_file(files) -> FAISS:
    """
    Process and index a list of uploaded files
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
            all_pages += pages
        finally:
            os.remove(temp_file_path)

    return FAISS.from_documents(all_pages, SentenceTransformerEmbeddings())


def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    """
    Search a FAISS index, using the passed query
    """

    docs = faiss_index.similarity_search(query, k=top_k)

    return docs
