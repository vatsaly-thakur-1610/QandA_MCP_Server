import os
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.ingestion import load_documents, build_chunks
from src.retrieval import build_retrievers
from src.graph import build_graph
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("PDF-Reader-Server")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
docs = load_documents()
chunks = build_chunks(docs)
faiss_retriever, bm25_retriever = build_retrievers(chunks, embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
app = build_graph(faiss_retriever, bm25_retriever, llm)

# --- 5. MCP TOOL ---
@mcp.tool()
def query_documents(question: str) -> str:
    """
    Query the indexed PDF documents to get grounded answers.
    IMPORTANT: Always reproduce the complete --- SOURCES CITED --- block from the tool response
    verbatim in your reply, including every document name and page number exactly as listed.
    Never paraphrase, summarise, or omit the sources section.
    """
    result = app.invoke({
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": ""
    })

    verdict = result.get("verdict", "")
    used_web = verdict in ("INCORRECT", "AMBIGUOUS") and result.get("web_docs")

    pdf_sources = sorted({
        f"- {os.path.basename(doc.metadata.get('source', 'Unknown'))} (Page {doc.metadata.get('page', 'N/A')})"
        for doc in result.get("good_docs", [])
    })
    web_sources = sorted({
        f"- {doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('url', 'N/A')})"
        for doc in result.get("web_docs", [])
    })

    footer = ""
    if used_web:
        footer += "\n> ⚠️ Note: PDF documents did not contain a confident answer — this response includes results from a web search.\n"
    if pdf_sources or web_sources:
        footer += "\n\n--- SOURCES CITED ---\n"
    if pdf_sources:
        footer += "**Document Sources:**\n" + "\n".join(pdf_sources) + "\n"
    if web_sources:
        footer += "**Web Sources:**\n" + "\n".join(web_sources) + "\n"
    if pdf_sources or web_sources:
        footer += "--- END SOURCES ---"

    output = (result["answer"] + footer) if footer else result["answer"]

    return output

@mcp.tool()
def list_documents() -> str:
    """
    List all PDF documents currently indexed in the knowledge base.
    """
    from src.ingestion import DOCS_DIR
    import glob
    pdfs = sorted(glob.glob(f"{DOCS_DIR}/*.pdf"))
    if not pdfs:
        return "No PDF documents found in the knowledge base."
    lines = [f"{i+1}. {os.path.basename(p)}" for i, p in enumerate(pdfs)]
    return f"Indexed documents ({len(pdfs)}):\n" + "\n".join(lines)


if __name__ == "__main__":
    mcp.run()