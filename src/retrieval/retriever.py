from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_retrievers(chunks: list[Document], embeddings: OpenAIEmbeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
    return faiss_retriever, bm25_retriever
