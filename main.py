# ============================================================
# AI Knowledge Base SaaS — Main Application
# Day 32 of 90 — Built by Asfer Saeed
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
import time
import os

from database import (
    init_db, get_db, KnowledgeBase,
    DocumentRecord, QueryRecord
)
from vector_store import (
    add_chunks, search_chunks,
    get_index_stats
)
from document_processor import (
    process_text_directly, chunk_text
)
from rag_engine import ask_question

load_dotenv()

# Initialize database
init_db()

app = FastAPI(
    title="AI Knowledge Base SaaS",
    description="""
## AI Knowledge Base SaaS
**Built by Asfer Saeed** — Day 32 of 90

Upload documents → Ask questions → Get AI answers

### Tech Stack
- FastAPI + PostgreSQL
- Pinecone vector search
- LangGraph RAG pipeline
- Groq LLM (llama-3.1-8b-instant)
    """,
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

# Add after app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# ── Request Models ────────────────────────────────────
class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="")

class DocCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=10)

class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

# ── Frontend ──────────────────────────────────────────
@app.get("/app", tags=["Frontend"])
async def serve_frontend():
    """Serve the frontend UI"""
    return FileResponse("static/index.html")

# ── Platform Endpoints ────────────────────────────────
@app.get("/", tags=["Platform"])
async def root(db: Session = Depends(get_db)):
    kb_count = db.query(KnowledgeBase).count()
    doc_count = db.query(DocumentRecord).count()
    query_count = db.query(QueryRecord).count()
    vector_stats = get_index_stats()

    return {
        "name": "AI Knowledge Base SaaS",
        "built_by": "Asfer Saeed",
        "day": "32 of 90",
        "stats": {
            "knowledge_bases": kb_count,
            "documents": doc_count,
            "queries_answered": query_count,
            "vectors_stored": vector_stats["total_vectors"]
        },
        "ui": "GET /app"
    }

# ── Knowledge Base Endpoints ──────────────────────────
@app.post("/kb", tags=["Knowledge Bases"])
async def create_kb(
    request: KBCreate,
    db: Session = Depends(get_db)
):
    """Create a new knowledge base"""
    kb = KnowledgeBase(
        name=request.name,
        description=request.description,
        pinecone_namespace=f"kb_{int(time.time())}"
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "created_at": str(kb.created_at)
    }

@app.get("/kb", tags=["Knowledge Bases"])
async def list_kbs(db: Session = Depends(get_db)):
    """List all knowledge bases"""
    kbs = db.query(KnowledgeBase).filter(
        KnowledgeBase.is_active == True
    ).all()
    return {
        "total": len(kbs),
        "knowledge_bases": [
            {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "doc_count": kb.doc_count,
                "query_count": kb.query_count,
                "created_at": str(kb.created_at)
            }
            for kb in kbs
        ]
    }

@app.get("/kb/{kb_id}", tags=["Knowledge Bases"])
async def get_kb(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Get knowledge base details"""
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id
    ).first()
    if not kb:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found"
        )
    docs = db.query(DocumentRecord).filter(
        DocumentRecord.kb_id == kb_id
    ).all()
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "doc_count": kb.doc_count,
        "query_count": kb.query_count,
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "chunk_count": d.chunk_count,
                "created_at": str(d.created_at)
            }
            for d in docs
        ]
    }

# ── Document Endpoints ────────────────────────────────

@app.post("/kb/{kb_id}/documents",
          tags=["Documents"])
async def add_document(
    kb_id: int,
    request: DocCreate,
    db: Session = Depends(get_db)
):
    """Add document to knowledge base"""
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id
    ).first()
    if not kb:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found"
        )

    # Process document
    processed = process_text_directly(
        request.title, request.content
    )

    # Save to PostgreSQL
    doc = DocumentRecord(
        kb_id=kb_id,
        filename=processed["filename"],
        title=request.title,
        content=request.content,
        chunk_count=processed["chunk_count"],
        file_type=processed["file_type"]
    )
    db.add(doc)
    db.flush()

    # Store chunks in Pinecone
    vectors_added = add_chunks(
        chunks=processed["chunks"],
        kb_id=kb_id,
        doc_id=doc.id
    )

    # Update KB stats
    kb.doc_count += 1
    db.commit()
    db.refresh(doc)

    return {
        "id": doc.id,
        "title": request.title,
        "kb_id": kb_id,
        "kb_name": kb.name,
        "chunks": processed["chunk_count"],
        "vectors_stored": vectors_added,
        "word_count": processed["word_count"]
    }

# ── File Upload Endpoint ──────────────────────────────
@app.post("/kb/{kb_id}/upload", tags=["Documents"])
async def upload_file(
    kb_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload PDF, TXT or Word file to knowledge base.
    AI automatically extracts text and indexes it.
    """
    # Validate KB exists
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id
    ).first()
    if not kb:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found"
        )

    # Validate file type
    filename = file.filename
    ext = filename.lower().split(".")[-1]
    allowed = ["pdf", "txt", "docx", "md", "csv"]

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{ext} not supported. "
                   f"Allowed: {allowed}"
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    print(f"📁 Uploading: {filename} "
          f"({file_size/1024:.1f} KB)")

    # Process file
    from document_processor import process_uploaded_file
    processed = process_uploaded_file(
        file_content=content,
        filename=filename
    )

    if not processed["chunks"]:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from file. "
                   "Please check the file content."
        )

    # Save to PostgreSQL
    doc = DocumentRecord(
        kb_id=kb_id,
        filename=filename,
        title=filename.rsplit(".", 1)[0],
        content=processed["text"][:5000],
        chunk_count=processed["chunk_count"],
        file_type=processed["file_type"],
        file_size=file_size
    )
    db.add(doc)
    db.flush()

    # Store chunks in Pinecone
    vectors_added = add_chunks(
        chunks=processed["chunks"],
        kb_id=kb_id,
        doc_id=doc.id
    )

    # Update KB stats
    kb.doc_count += 1
    db.commit()
    db.refresh(doc)

    return {
        "message": "File uploaded and indexed successfully!",
        "document_id": doc.id,
        "filename": filename,
        "file_type": processed["file_type"],
        "file_size_kb": round(file_size / 1024, 1),
        "pages_or_paragraphs": processed["word_count"],
        "chunks_created": processed["chunk_count"],
        "vectors_stored": vectors_added,
        "kb_name": kb.name,
        "ready_to_query": True
    }

@app.get("/kb/{kb_id}/documents",
         tags=["Documents"])
async def list_documents(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """List documents in knowledge base"""
    docs = db.query(DocumentRecord).filter(
        DocumentRecord.kb_id == kb_id
    ).all()
    return {
        "kb_id": kb_id,
        "total": len(docs),
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "chunk_count": d.chunk_count,
                "word_count": len(
                    d.content.split()
                ) if d.content else 0,
                "created_at": str(d.created_at)
            }
            for d in docs
        ]
    }

# ── AI Question Endpoint ──────────────────────────────
@app.post("/kb/{kb_id}/ask", tags=["AI Agent"])
async def ask(
    kb_id: int,
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """Ask AI a question about the knowledge base"""
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id
    ).first()
    if not kb:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found"
        )

    # Check documents exist
    doc_count = db.query(DocumentRecord).filter(
        DocumentRecord.kb_id == kb_id
    ).count()
    if doc_count == 0:
        return {
            "answer": "This knowledge base has no documents yet. "
                     "Please add some documents first!",
            "confidence": 0.0,
            "sources_count": 0,
            "latency_ms": 0
        }

    # Run RAG pipeline
    result = ask_question(
        question=request.question,
        kb_id=kb_id
    )

    # Log to PostgreSQL
    query_log = QueryRecord(
        kb_id=kb_id,
        question=request.question,
        answer=result["answer"],
        sources_count=result["sources_count"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"]
    )
    db.add(query_log)
    kb.query_count += 1
    db.commit()

    return {
        "kb_name": kb.name,
        "question": request.question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources_count": result["sources_count"],
        "latency_ms": result["latency_ms"],
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

# ── Stats ─────────────────────────────────────────────
@app.get("/stats", tags=["Platform"])
async def stats(db: Session = Depends(get_db)):
    """Platform statistics"""
    vector_stats = get_index_stats()
    return {
        "knowledge_bases": db.query(
            KnowledgeBase
        ).count(),
        "documents": db.query(
            DocumentRecord
        ).count(),
        "queries_answered": db.query(
            QueryRecord
        ).count(),
        "vectors_stored": vector_stats["total_vectors"],
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

@app.get("/health", tags=["Platform"])
async def health():
    """Health check for Railway deployment"""
    return {
        "status": "healthy",
        "service": "AI Knowledge Base SaaS",
        "built_by": "Asfer Saeed",
        "day": "34 of 90",
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }