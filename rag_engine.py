# ============================================================
# AI Knowledge Base SaaS — RAG Engine
# LangGraph + Pinecone + Groq
# Built by: Asfer Saeed
# ============================================================

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from typing import TypedDict, List
import time
import os

from vector_store import search_chunks

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── RAG State ─────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    kb_id: int
    chunks: List[dict]
    context: str
    answer: str
    confidence: float
    sources_count: int
    attempts: int

# ── Nodes ─────────────────────────────────────────────
def search_node(state: RAGState) -> dict:
    """Search Pinecone for relevant chunks"""
    chunks = search_chunks(
        query=state["question"],
        kb_id=state["kb_id"],
        top_k=5
    )
    context = "\n\n".join([
        f"[Source {i+1}]: {c['text']}"
        for i, c in enumerate(chunks)
    ])
    return {
        "chunks": chunks,
        "context": context,
        "attempts": state["attempts"] + 1
    }

def generate_node(state: RAGState) -> dict:
    """Generate answer from retrieved context"""
    if not state["context"]:
        return {
            "answer": "I don't have enough information "
                     "to answer this question.",
            "confidence": 0.0,
            "sources_count": 0
        }

    response = llm.invoke([
        SystemMessage(content=
            "You are a helpful knowledge base assistant. "
            "Answer using ONLY the provided context. "
            "If the answer is not in the context, say "
            "'I don't have that information in this "
            "knowledge base.' "
            "Be concise and accurate."
        ),
        HumanMessage(content=
            f"Context:\n{state['context']}\n\n"
            f"Question: {state['question']}"
        )
    ])

    return {
        "answer": response.content,
        "sources_count": len(state["chunks"])
    }

def evaluate_node(state: RAGState) -> dict:
    """Evaluate answer confidence"""
    if not state["answer"] or \
       "don't have" in state["answer"].lower():
        return {"confidence": 0.3}

    response = llm.invoke([
        SystemMessage(content=
            "Rate answer quality 0.0-1.0. "
            "Return ONLY a decimal number."
        ),
        HumanMessage(content=
            f"Q: {state['question']}\n"
            f"A: {state['answer']}"
        )
    ])

    try:
        confidence = float(
            response.content.strip()
        )
        confidence = max(0.0, min(1.0, confidence))
    except:
        confidence = 0.75

    return {"confidence": confidence}

# ── Routing ───────────────────────────────────────────
def route(state: RAGState) -> str:
    if state["attempts"] >= 2:
        return "end"
    if not state["chunks"]:
        return "end"
    return "evaluate"

def route_after_eval(state: RAGState) -> str:
    if state["confidence"] >= 0.5:
        return "end"
    if state["attempts"] >= 2:
        return "end"
    return "search"

# ── Build Graph ───────────────────────────────────────
def build_rag_graph():
    g = StateGraph(RAGState)
    g.add_node("search", search_node)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)

    g.set_entry_point("search")
    g.add_edge("search", "generate")
    g.add_conditional_edges(
        "generate", route,
        {"evaluate": "evaluate", "end": END}
    )
    g.add_conditional_edges(
        "evaluate", route_after_eval,
        {"end": END, "search": "search"}
    )
    return g.compile()

rag_graph = build_rag_graph()

def ask_question(
    question: str,
    kb_id: int
) -> dict:
    """Run RAG pipeline for a question"""
    start = time.time()
    result = rag_graph.invoke({
        "question": question,
        "kb_id": kb_id,
        "chunks": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "sources_count": 0,
        "attempts": 0
    })
    latency = time.time() - start

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources_count": result["sources_count"],
        "latency_ms": round(latency * 1000)
    }