import os
import re
import uuid
import subprocess, sys
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

import numpy as np
import fitz  # PyMuPDF

from src.db import ensure_tables, last_runs

# =========================
# Config
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "semantic_collection")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
VECTOR_SIZE = 384
DISTANCE = models.Distance.COSINE

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=180.0)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# =========================
# FastAPI
# =========================
app = FastAPI(title="HappyRuH + Qdrant Semantic Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# Helpers
# =========================
def ensure_collection_exists():
    names = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )

def chunk_text(text: str, words: int = 120) -> List[str]:
    w = text.split()
    return [" ".join(w[i:i+words]) for i in range(0, len(w), words)] if w else []

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")
    out = []
    for page in doc:
        raw = page.get_text("text") or ""
        lines = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if len(s) >= 10:
                uppercase_ratio = sum(1 for c in s if c.isupper()) / max(1, len(s))
                if uppercase_ratio > 0.6 and re.match(r'^[A-Z0-9\s\-\,:()]+$', s):
                    continue
            if re.match(r"^Page\s*\d+", s, re.IGNORECASE):
                continue
            lines.append(s)
        if lines:
            out.append("\n".join(lines))
    text = "\n".join(out)
    return re.sub(r"\n{2,}", "\n", text).strip()

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32)
    embs = embedder.encode(texts, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)

# =========================
# Schemas
# =========================
class Product(BaseModel):
    name: str
    price: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = ""
    image: Optional[str] = None
    category: Optional[str] = None

class IngestProductsRequest(BaseModel):
    products: List[Product]

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    _ = qdrant.get_collections()
    return {"status": "ok", "qdrant": QDRANT_URL, "collection": QDRANT_COLLECTION, "qdrant_api_key_enabled": bool(QDRANT_API_KEY)}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), source_label: str = "pdf_upload"):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    text = extract_text_from_pdf(content)
    if not text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not create chunks.")
    ensure_collection_exists()
    vectors = embed_texts(chunks)
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i].tolist(),
            payload={
                "type": "pdf_chunk",
                "source": source_label,
                "source_filename": file.filename,
                "chunk_index": i,
                "text": chunks[i],
            }
        ) for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return {"status": "success", "file": file.filename, "added_chunks": len(points)}

@app.post("/ingest-products")
def ingest_products(body: IngestProductsRequest):
    if not body.products:
        raise HTTPException(status_code=400, detail="No products provided.")
    ensure_collection_exists()
    texts, payloads = [], []
    for p in body.products:
        t = (
            f"Product Name: {p.name}. "
            f"Price: {p.price or 'N/A'}. "
            f"Description: {p.description or ''}. "
            f"URL: {p.url or ''}. Image: {p.image or ''}. "
            f"Category: {p.category or ''}."
        )
        texts.append(t)
        payloads.append({
            "type": "product",
            "source": "happyruh_scraper",
            "text": t,
            "product_name": p.name,
            "price": p.price,
            "url": p.url,
            "image": p.image,
            "category": p.category,
        })
    vectors = embed_texts(texts)
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i].tolist(),
            payload=payloads[i]
        ) for i in range(len(texts))
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return {"status": "success", "inserted": len(points)}

@app.get("/search")
def search(q: str = Query(..., description="Your query"), top_k: int = 5):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    ensure_collection_exists()
    qvec = embed_texts([q])[0].tolist()
    hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=max(1, min(top_k, 50)))
    matches = []
    for h in hits or []:
        p = h.payload or {}
        matches.append({
            "score": float(h.score),
            "type": p.get("type"),
            "source": p.get("source"),
            "product_name": p.get("product_name"),
            "price": p.get("price"),
            "url": p.get("url"),
            "image": p.get("image"),
            "category": p.get("category"),
            "source_filename": p.get("source_filename"),
            "chunk_index": p.get("chunk_index"),
            "text_snippet": (p.get("text") or "")[:400],
        })
    return {"query": q, "matches": matches}

# ---- pipeline controls ----
@app.post("/sync-now")
def sync_now():
    """Run the pipeline immediately: scrape → Postgres → clean+embed → Qdrant."""
    try:
        ensure_tables()
        r1 = subprocess.run([sys.executable, "src/happyruh_scraper.py"], capture_output=True, text=True)
        if r1.returncode != 0:
            raise RuntimeError(f"scraper failed: {r1.stderr or r1.stdout}")
        r2 = subprocess.run([sys.executable, "src/semantic_search_loader.py"], capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"loader failed: {r2.stderr or r2.stdout}")
        return {"status": "ok", "scraper_out_tail": r1.stdout[-2000:], "loader_out_tail": r2.stdout[-2000:]}
    except Exception as e:
        raise HTTPException(500, f"sync failed: {e}")

@app.get("/pipeline-status")
def pipeline_status(limit: int = 10):
    """Show recent pipeline runs with counts & status."""
    try:
        runs = last_runs(limit)
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(500, f"status failed: {e}")
