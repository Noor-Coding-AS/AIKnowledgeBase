# ============================================================
# AI Knowledge Base SaaS — Database
# Built by: Asfer Saeed
# ============================================================

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Text, DateTime, Float, Boolean, ForeignKey
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship
)
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

# DATABASE_URL = os.getenv(
#     "POSTGRES_URL",
#     "postgresql://postgres:admin%40123@localhost:5432/aiknowledgebase"
# )

DATABASE_URL = os.getenv(
    "DATABASE_URL",  # Railway sets this automatically
    os.getenv(
        "POSTGRES_URL",
        "postgresql://postgres:admin%40123@localhost:5432/aiknowledgebase"
    )
)

# Railway uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql://", 1
    )

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ── Models ────────────────────────────────────────────
class KnowledgeBase(Base):
    __tablename__ = "kb_saas"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, default="")
    owner = Column(String(100), default="default")
    pinecone_namespace = Column(String(200))
    doc_count = Column(Integer, default=0)
    query_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    documents = relationship(
        "DocumentRecord",
        back_populates="kb"
    )
    queries = relationship(
        "QueryRecord",
        back_populates="kb"
    )

class DocumentRecord(Base):
    __tablename__ = "documents_saas"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("kb_saas.id"))
    filename = Column(String(300))
    title = Column(String(300))
    content = Column(Text)
    chunk_count = Column(Integer, default=0)
    file_type = Column(String(20), default="text")
    file_size = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    kb = relationship("KnowledgeBase",
                      back_populates="documents")

class QueryRecord(Base):
    __tablename__ = "queries_saas"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("kb_saas.id"))
    question = Column(Text)
    answer = Column(Text)
    sources_count = Column(Integer, default=0)
    confidence = Column(Float, default=0.0)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    kb = relationship("KnowledgeBase",
                      back_populates="queries")

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created!")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()