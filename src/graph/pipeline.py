import re
from typing import List, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from src.prompts import answer_prompt, doc_eval_prompt, filter_prompt, rewrite_prompt

UPPER_TH = 0.7
LOWER_TH = 0.3


class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str


class DocEvalScore(BaseModel):
    score: float
    reason: str


class KeepOrDrop(BaseModel):
    keep: bool


class WebQuery(BaseModel):
    query: str


def build_graph(faiss_retriever, bm25_retriever, llm: ChatOpenAI):
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
    filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)
    rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)
    tavily = TavilySearchResults(max_results=5)

    def retrieve_node(state: State) -> State:
        q = state["question"]
        faiss_docs = faiss_retriever.invoke(q)
        bm25_docs = bm25_retriever.invoke(q)
        seen, merged = set(), []
        for doc in faiss_docs + bm25_docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:100])
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return {"docs": merged[:8]}

    def eval_each_doc_node(state: State) -> State:
        q = state["question"]
        scores: List[float] = []
        good: List[Document] = []
        for d in state["docs"]:
            out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
            scores.append(out.score)
            if out.score > LOWER_TH:
                good.append(d)
        if any(s > UPPER_TH for s in scores):
            return {"good_docs": good, "verdict": "CORRECT", "reason": f"At least one retrieved chunk scored > {UPPER_TH}."}
        if len(scores) > 0 and all(s < LOWER_TH for s in scores):
            return {"good_docs": [], "verdict": "INCORRECT", "reason": f"All retrieved chunks scored < {LOWER_TH}."}
        return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}."}

    def _decompose_to_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def refine(state: State) -> State:
        q = state["question"]
        if state.get("verdict") == "CORRECT":
            docs_to_use = state["good_docs"]
        elif state.get("verdict") == "INCORRECT":
            docs_to_use = state["web_docs"]
        else:
            docs_to_use = state["good_docs"] + state["web_docs"]
        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        strips = _decompose_to_sentences(context)
        kept = [s for s in strips if filter_chain.invoke({"question": q, "sentence": s}).keep]
        return {"strips": strips, "kept_strips": kept, "refined_context": "\n".join(kept).strip()}

    def rewrite_query_node(state: State) -> State:
        out = rewrite_chain.invoke({"question": state["question"]})
        return {"web_query": out.query}

    def web_search_node(state: State) -> State:
        q = state.get("web_query") or state["question"]
        results = tavily.invoke({"query": q})
        web_docs: List[Document] = []
        for r in results or []:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        return {"web_docs": web_docs}

    def generate(state: State) -> State:
        out = (answer_prompt | llm).invoke({"question": state["question"], "context": state["refined_context"]})
        return {"answer": out.content}

    def route_after_eval(state: State) -> str:
        return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"

    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges("eval_each_doc", route_after_eval, {"refine": "refine", "rewrite_query": "rewrite_query"})
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()
