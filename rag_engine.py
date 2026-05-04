# ============================================================
# AI Knowledge Base SaaS — RAG Engine (Simple + Reliable)
# No LangGraph — direct Groq call
# Built by: Asfer Saeed — Day 35
# ============================================================

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import time
import os

load_dotenv()

def ask_question(question: str, kb_id: int) -> dict:
    """Simple RAG pipeline — no LangGraph needed"""
    start = time.time()

    # Step 1 — Search Pinecone
    try:
        from vector_store import search_chunks
        chunks = search_chunks(
            query=question,
            kb_id=kb_id,
            top_k=5
        )
    except Exception as e:
        print(f"Search error: {e}")
        chunks = []

    # Step 2 — Build context
    if chunks:
        context = "\n\n".join([
            f"Source {i+1}: {c['text']}"
            for i, c in enumerate(chunks)
        ])
        confidence = 0.8
    else:
        context = "No documents found in this knowledge base."
        confidence = 0.2

    # Step 3 — Generate answer
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=api_key
        )

        response = llm.invoke([
            SystemMessage(content=
                "You are a helpful assistant. "
                "Answer using ONLY the provided context. "
                "If not in context say: "
                "'I dont have that information.' "
                "Be concise."
            ),
            HumanMessage(content=
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        ])
        answer = response.content

    except Exception as e:
        print(f"LLM error: {e}")
        answer = (
            f"Error: {str(e)[:100]}. "
            "Check GROQ_API_KEY in Railway variables."
        )
        confidence = 0.0

    latency = time.time() - start
    return {
        "answer": answer,
        "confidence": confidence,
        "sources_count": len(chunks),
        "latency_ms": round(latency * 1000)
    }