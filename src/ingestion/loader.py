import glob
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "documents")


def load_documents(docs_dir: str = DOCS_DIR) -> list[Document]:
    docs = []
    for pdf_path in sorted(glob.glob(f"{docs_dir}/*.pdf")):
        docs += PyPDFLoader(pdf_path).load()
    return docs


def build_chunks(docs: list[Document]) -> list[Document]:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return chunks
