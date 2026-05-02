# ============================================================
# AI Knowledge Base SaaS — Vector Store
# Built by: Asfer Saeed
# ============================================================

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import time
import uuid

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "asfer-ai-knowledge-base"
DIMENSION = 384

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Creating Pinecone index: {INDEX_NAME}")
    time.sleep(15)

index = pc.Index(INDEX_NAME)
print(f"✅ Pinecone ready: {INDEX_NAME}")

def add_chunks(
    chunks: list,
    kb_id: int,
    doc_id: int,
    namespace: str = "default"
) -> int:
    """Add document chunks to Pinecone"""
    if not chunks:
        return 0

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        vector_id = f"kb{kb_id}_doc{doc_id}_chunk{i}"
        vectors.append((
            vector_id,
            embedding,
            {
                "text": chunk[:500],
                "kb_id": kb_id,
                "doc_id": doc_id,
                "chunk_index": i
            }
        ))

    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    return len(vectors)

def search_chunks(
    query: str,
    kb_id: int,
    top_k: int = 5
) -> list:
    """Search Pinecone for relevant chunks"""
    query_vector = embedder.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"kb_id": kb_id}
    )

    return [
        {
            "text": match.metadata.get("text", ""),
            "score": match.score,
            "doc_id": match.metadata.get("doc_id")
        }
        for match in results.matches
    ]

def delete_kb_vectors(kb_id: int):
    """Delete all vectors for a knowledge base"""
    try:
        index.delete(filter={"kb_id": kb_id})
        print(f"✅ Deleted vectors for KB {kb_id}")
    except Exception as e:
        print(f"Delete error: {e}")

def get_index_stats() -> dict:
    """Get Pinecone index statistics"""
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "index_name": INDEX_NAME,
        "dimension": DIMENSION
    }