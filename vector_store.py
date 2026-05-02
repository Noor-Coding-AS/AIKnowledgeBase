# ============================================================
# AI Knowledge Base SaaS — Vector Store (Lazy Init)
# Built by: Asfer Saeed
# ============================================================

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
import math
import hashlib

load_dotenv()

INDEX_NAME = "ai-knowledge-base"
DIMENSION = 1536

# ── Lazy initialization — no crash on missing key ─────
_index = None

def get_index():
    """Get Pinecone index — lazy init"""
    global _index
    if _index is not None:
        return _index

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not set. "
            "Add it to Railway environment variables!"
        )

    pc = Pinecone(api_key=api_key)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", region="us-east-1"
            )
        )
        print(f"Creating index: {INDEX_NAME}")
        time.sleep(15)

    _index = pc.Index(INDEX_NAME)
    print(f"✅ Pinecone ready: {INDEX_NAME}")
    return _index

# ── Lightweight embedding ─────────────────────────────
def simple_embed(text: str) -> list:
    """Hash-based embedding — no PyTorch needed"""
    vector = []
    for i in range(DIMENSION):
        hash_input = f"{text[:100]}_{i}".encode()
        h = hashlib.md5(hash_input).hexdigest()
        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1
        vector.append(val)
    magnitude = math.sqrt(sum(v*v for v in vector))
    if magnitude > 0:
        vector = [v/magnitude for v in vector]
    return vector

def add_chunks(chunks: list, kb_id: int,
               doc_id: int, namespace: str = "default") -> int:
    if not chunks:
        return 0
    index = get_index()
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = simple_embed(chunk)
        vector_id = f"kb{kb_id}_doc{doc_id}_chunk{i}"
        vectors.append((
            vector_id, embedding,
            {"text": chunk[:500], "kb_id": kb_id,
             "doc_id": doc_id, "chunk_index": i}
        ))
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])
    return len(vectors)

def search_chunks(query: str, kb_id: int,
                  top_k: int = 5) -> list:
    index = get_index()
    query_vector = simple_embed(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"kb_id": kb_id}
    )
    return [
        {
            "text": m.metadata.get("text", ""),
            "score": m.score,
            "doc_id": m.metadata.get("doc_id")
        }
        for m in results.matches
    ]

def delete_kb_vectors(kb_id: int):
    try:
        get_index().delete(filter={"kb_id": kb_id})
    except Exception as e:
        print(f"Delete error: {e}")

def get_index_stats() -> dict:
    try:
        stats = get_index().describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "index_name": INDEX_NAME,
            "dimension": DIMENSION
        }
    except Exception as e:
        return {"total_vectors": 0,
                "index_name": INDEX_NAME,
                "error": str(e)}